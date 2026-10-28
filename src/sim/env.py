"""SpaceXEnv: the booster's landing burn as a Gymnasium environment on our own physics (no Box2D).

Observation (18, all roughly in [-1, 1]):
   0 x / 2000 m      pad-relative horizontal position        7 thrust level (fraction of full thrust, actual)
   1 h / 3000 m      foot height above the pad               8 gimbal / max
   2 vx / 300 m/s                                            9 fin deflection / max
   3 vy / 300 m/s    (negative = falling)                   10 RCS gas left (0..1)
   4 tilt (rad)      0 = upright                            11 engine running (0/1)
   5 spin (rad/s) / 0.5                                      12 ignitions used / allowed
   6 propellant left (0..1 of the loaded mass)              13 dynamic pressure / 50 kPa
  14-17: the same x, h, vx, vy again on a square-root scale (sign kept): sqrt(|v|/scale) gives the last metres and the last m/s of a
  landing a resolution the linear channels do not have (h = 10 m is 0.003 linear, 0.06 here).
Action (5, each in [-1, 1]):  throttle (<= -0.5 engine off, above: 40..100%), gimbal, fin, RCS system A, RCS system B.

The rules of the old project are built in (and staged): tilt limit, ignition limit, descent clock, stall rule, fuel (real propellant mass).
Touchdown ends the episode: 'rest' (survived) or 'crash'.
"""
from __future__ import annotations

import bisect
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.sim import vehicle as V
from src.sim.physics import State, min_landing_time, step
from src.sim.spawn import Wind, make_spawn
from src.training import merit as M
from src.training import reward as R
from src.utils import config as C

OBS_DIM = 18
ACT_DIM = 5
X_LIMIT_M = 9000.0                  # leaving this far from the pad (or this high) ends the episode as 'oob'
Y_LIMIT_M = 6000.0
ON_THRESHOLD = -0.5                 # throttle action at or below this = engine off
BAND_EDGES = C.ALT_BANDS_M


def _ssqrt(v: float) -> float:
    """sign(v) * sqrt(|v|)"""
    return math.copysign(math.sqrt(abs(v)), v)


def tilt_from_vertical(angle_rad: float) -> float:
    return abs(math.remainder(float(angle_rad), 2.0 * math.pi))


def throttle_from_action(a0: float) -> float:
    """-1..1 -> 0 (off) or the 40..100% range of the engine."""
    if a0 <= ON_THRESHOLD:
        return 0.0
    return V.THROTTLE_MIN + (1.0 - V.THROTTLE_MIN) * min(1.0, (a0 - ON_THRESHOLD) / (1.0 - ON_THRESHOLD))


def vehicle_description() -> dict:
    """What the viewer needs to draw the booster to scale and label the numbers."""
    return dict(dry=V.DRY_MASS, length=V.LENGTH, diameter=V.DIAMETER, cg=V.CG_BASE, foot=V.FOOT_BELOW_BASE, span=V.LEG_SPAN, fin_arm=V.FIN_ARM,
                rcs_a=V.RCS_ARM_A, rcs_b=V.RCS_ARM_B, gimbal_max=V.GIMBAL_MAX, fin_max=V.FIN_MAX, thrust_sl=V.THRUST_SL, thrust_vac=V.THRUST_VAC,
                throttle_min=V.THROTTLE_MIN, mdot=V.MDOT_FULL, rcs_force=V.RCS_FORCE, g=C.GRAVITY_MPS2, leg_down=C.LEG_DOWN_M,
                pad=[-C.PAD_HALF_WIDTH_M, C.PAD_HALF_WIDTH_M, 0.0], brake_max=C.BRAKE_MAX_MPS2, budget=C.FUEL_BUDGET, zone=C.SUCCESS_MAX_DIST_M,
                soft_speed=C.SUCCESS_MAX_SPEED_MPS, env_decel=R.ENVELOPE_DECEL, env_floor=R.ENVELOPE_FLOOR, tilt_limit_deg=C.TILT_LIMIT_DEG, fps=C.FPS)


class SpaceXEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, level: int = 0, fuel_budget: float | None = None, record: bool = False, randomize: bool = True):
        super().__init__()
        self.level = int(level)
        self.budget = C.FUEL_BUDGET if fuel_budget is None else float(fuel_budget)
        if not math.isfinite(self.budget):
            self.budget = 1e9
        self.record = record
        self.rules_stage = C.RULES_STAGE_START
        self.pending_stage = None
        self.observation_space = spaces.Box(-20.0, 20.0, (OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), dtype=np.float32)
        self.s: State | None = None
        self._clear()

    # ------------------------------------------------------------------ rules staging / curriculum (called through env_method)
    def set_level(self, level: int) -> None:
        self.level = int(level)                      # takes effect on the next reset

    def set_rules_stage(self, stage: int) -> None:
        self.pending_stage = int(stage)              # takes effect on the next reset, so no episode changes its rules half way through

    def _clear(self):
        self.t = 0
        self.frames: list = []
        self.terms_sum: dict = {}
        self.td = None
        self.pending_td = None
        self.ascended = False
        self.prev_state = None
        self.prev_fuel = 0.0
        self.air_steps = self.env_ok_steps = 0
        self.brake_peak = 0.0
        self.band_sum = [0.0] * (len(BAND_EDGES) + 1)
        self.band_n = [0] * (len(BAND_EDGES) + 1)
        self.down = False
        self.deadline = self.limit_s = 0.0
        self.wind = Wind(0.0)
        self.out_of_fuel_reported = False
        self.settle_end = 0
        self.settled_steps = 0

    # ------------------------------------------------------------------ observation / reward state
    def _obs(self) -> np.ndarray:
        s = self.s
        h = max(s.foot_height(), 0.0)
        gas_frac = s.gas / V.RCS_GAS
        o = [s.x / C.OBS_X_TO_M, h / C.OBS_Y_TO_M, s.vx / C.OBS_VX_TO_MPS, s.vy / C.OBS_VY_TO_MPS, s.th, s.om / C.OBS_SPIN_TO_RAD_S,
             s.prop / self.budget, s.thr, s.gim / V.GIMBAL_MAX, s.fin / V.FIN_MAX, gas_frac, 1.0 if s.on else 0.0,
             min(1.0, s.ign / float(C.MAX_IGNITIONS if C.MAX_IGNITIONS > 0 else 10)), s.q / C.OBS_Q_TO_PA]
        o += [_ssqrt(o[0]), math.sqrt(max(o[1], 0.0)), _ssqrt(o[2]), _ssqrt(o[3])]
        return np.clip(np.asarray(o, dtype=np.float32), -20.0, 20.0)

    def _rstate(self, legs: int = 0) -> R.State:
        s = self.s
        h = max(s.foot_height(), 0.0)
        return R.State(dist=math.hypot(s.x, h), speed=math.hypot(s.vx, s.vy), height=h, angle=s.th, spin=s.om, legs=legs, vy=s.vy, x=s.x, vx=s.vx)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.pending_stage is not None:
            self.rules_stage, self.pending_stage = self.pending_stage, None
        self._clear()
        cfg = C.LEVELS[self.level]
        opt = options or {}
        if "spawn" in opt:                           # a start handed in by a wrapper (the dual-lander environment): (State, details) and the shared wind
            self.wind = opt.get("wind") or Wind(0.0, self.np_random)
            self.s, self.spawn = opt["spawn"][0].copy(), dict(opt["spawn"][1])
            self.s.prop = self.budget
        else:
            self.wind = Wind(cfg["wind_power"], self.np_random)
            self.s, self.spawn = make_spawn(self.np_random, self.level, prop=self.budget, wind=self.wind)
        s = self.s
        h = s.foot_height()
        fastest = min_landing_time(h, -s.vy, C.BRAKE_MAX_MPS2, C.GRAVITY_MPS2, C.SUCCESS_MAX_SPEED_MPS)
        self.limit_s = max(C.TIME_LIMIT_S, C.TIME_LIMIT_SCALE * fastest + C.TIME_LIMIT_PAD_S)
        self.deadline = int(round(self.limit_s * C.FPS))
        self.prev_state = self._rstate()
        info = dict(level=self.level, rules_stage=self.rules_stage, time_limit_s=self.limit_s, fastest_landing_s=fastest, spawn=self.spawn)
        return self._obs(), info

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(ACT_DIM), -1.0, 1.0)
        s = self.s
        # ---- rules that act on the command
        throttle = throttle_from_action(float(a[0]))
        ignition = blocked = False
        if throttle > 0.0 and not s.on and not s.out_of_prop:
            if self.rules_stage >= 2 and C.MAX_IGNITIONS > 0 and s.ign >= C.MAX_IGNITIONS:
                throttle, blocked = 0.0, True                 # no ignitions left: the command is ignored
            else:
                ignition = True
        prop0, gas0 = s.prop, s.gas
        gim, fin = float(a[1]) * V.GIMBAL_MAX, float(a[2]) * V.FIN_MAX
        # ---- physics
        dt = 1.0 / (C.FPS * C.PHYSICS_SUBSTEPS)
        if self.td is not None and C.TOUCHDOWN_ENGINE_CUT:             # the engine shut down at touchdown and stays off while the lander settles
            throttle, ignition, blocked = 0.0, False, False
        first_v = None                                                   # the speed at the instant the first foot touches
        for _ in range(C.PHYSICS_SUBSTEPS):
            pre_v = (s.vx, s.vy)
            w = self.wind.at(s.t, s.y)
            step(s, throttle, gim, fin, float(a[3]), float(a[4]), dt, w)
            if first_v is None and self.td is None and s.foot_height() <= 0.0:
                first_v = pre_v
                if C.TOUCHDOWN_ENGINE_CUT:
                    throttle, ignition = 0.0, False
        self.t += 1
        ignitions = s.ign
        out_of_fuel_event = s.out_of_prop and not self.out_of_fuel_reported
        self.out_of_fuel_reported = self.out_of_fuel_reported or s.out_of_prop
        # ---- touchdown / termination
        td = None
        if first_v is not None:
            td = dict(dist_m=abs(s.x), speed_mps=math.hypot(*first_v), tilt_rad=abs(s.th), step=self.t, vspeed_mps=abs(first_v[1]), hspeed_mps=abs(first_v[0]))
            self.td = self.pending_td = td
            self.settle_end = self.t + int(round(C.SETTLE_S * C.FPS))
            self.settled_steps = 0
        terminated = truncated = False
        end = None
        tilt = tilt_from_vertical(s.th)
        h = s.foot_height()
        speed_now = math.hypot(s.vx, s.vy)
        if self.td is not None:                                          # on the ground: judge the impact, then the settling (see config SETTLE_S)
            end = self._settle_verdict(td)
            terminated = end is not None
        elif tilt > math.radians(C.TILT_LIMIT_DEG):
            end, terminated = "tilt", True
        elif (self.rules_stage >= 1 and self.t > C.STALL_GRACE_STEPS and h > C.STALL_MIN_HEIGHT_M and speed_now < C.STALL_SPEED_MPS):
            end, terminated = "stall", True
        elif abs(s.x) >= X_LIMIT_M or s.y >= Y_LIMIT_M:
            end, terminated = "oob", True
        elif (self.rules_stage >= 1 and self.t >= self.deadline) or self.t >= C.MAX_EPISODE_STEPS:
            end, truncated = "timeout", True

        # ---- metrics that need every step
        if self.td is None:
            hb = max(h, 0.0)
            b = bisect.bisect(BAND_EDGES, hb)
            self.band_sum[b] += min(4.0, speed_now / max(R.envelope_speed(hb), 1e-3))
            self.band_n[b] += 1
            if h > 100.0:
                self.air_steps += 1
                self.env_ok_steps += speed_now <= R.envelope_speed(h)
                vd = max(0.0, -s.vy)
                self.brake_peak = max(self.brake_peak, vd * vd / (2.0 * h) / C.BRAKE_MAX_MPS2)

        # ---- reward
        cur = self._rstate(legs=2 if end == "rest" else int(s.contact) if self.td is not None else 0)
        ascend = (not self.ascended) and R.is_ascending(cur)
        self.ascended = self.ascended or ascend
        arrival = None
        if end in ("crash", "oob"):
            arrival = self.pending_td or dict(dist_m=abs(self.prev_state.x), tilt_rad=abs(self.prev_state.angle), speed_mps=self.prev_state.speed)
        elif end in ("tilt", "stall"):
            arrival = self.pending_td or R.projected_impact(self.prev_state)
        burned_units = (prop0 - s.prop) / (self.budget * 0.01) if self.budget < 1e8 else 0.0
        total, terms = R.compute_reward(self.prev_state, cur, burned_units, end=end if (terminated or truncated) else None,
                                        fuel_out_event=out_of_fuel_event, landing=self.td if end == "rest" else None, ascend_event=ascend, arrival=arrival,
                                        ignition=ignitions if ignition else 0, refire_on=self.rules_stage >= 2)
        self.prev_state = cur
        for k, v in terms.items():
            self.terms_sum[k] = self.terms_sum.get(k, 0.0) + v

        info = dict(touchdown=self.td, rules_stage=self.rules_stage, ignitions=ignitions, ignition=ignition, ignition_blocked=blocked,
                    fuel_used=self.budget - s.prop if self.budget < 1e8 else 0.0, fuel_remaining=(s.prop / self.budget if self.budget < 1e8 else 1.0),
                    fuel_out_event=out_of_fuel_event, out_of_fuel=s.out_of_prop, reward_terms=terms, reward_raw=total, time_s=self.t / C.FPS,
                    time_limit_s=self.limit_s, tilt_deg=math.degrees(tilt))
        if end == "tilt":
            info["tilt_out"] = True
        if end == "stall":
            info["stall_out"] = True
        if end == "timeout":
            info["time_out"] = True
        if self.record:
            self.frames.append(self._frame(terms, a, throttle, self.td is not None, end == "rest"))
        if terminated or truncated:
            info["episode_metrics"] = self._metrics(end, tilt, h)
            if self.record:
                m = info["episode_metrics"]
                info["trajectory"] = dict(
                    terrain=[[-X_LIMIT_M, 0.0], [X_LIMIT_M, 0.0]], pad=[-C.PAD_HALF_WIDTH_M, C.PAD_HALF_WIDTH_M, 0.0], dt=1.0 / C.FPS, veh=vehicle_description(),
                    frames=self.frames, level=m["level"], stage=m["rules_stage"],
                    result=dict(end=m["end"], success=bool(m["success"]), soft=bool(m["soft"]), merit=round(float(m["merit"]), 1),
                                speed=None if m["speed_mps"] != m["speed_mps"] else round(float(m["speed_mps"]), 3), dist=round(float(m["dist_m"]), 3),
                                fuel_out=bool(m["out_of_fuel"]), score=round(sum(sum(f["t"].values()) for f in self.frames), 1)))
        return self._obs(), float(total * R.GLOBAL_SCALE), terminated, truncated, info

    # ------------------------------------------------------------------ outputs
    def _frame(self, terms, a, throttle, touched, rest) -> dict:
        s = self.s
        r3 = lambda v: round(float(v), 3)
        (xa, ya), (xb, yb) = s.feet()
        return dict(x=r3(s.x), y=r3(s.y), a=r3(s.th), legs=[[r3(xa), r3(ya), 0.0, int(s.pen_a > 0)], [r3(xb), r3(yb), 0.0, int(s.pen_b > 0)]], lc=[r3(s.pen_a), r3(s.pen_b)], m=round(s.thr, 3),
                    s=round(max(abs(s.cmd_a), abs(s.cmd_b)), 2), side=1 if (s.cmd_a + s.cmd_b) > 0 else -1, fuel=r3(s.prop / self.budget if self.budget < 1e8 else 1.0),
                    g=r3(s.gim), f=r3(s.fin), r1=round(s.cmd_a, 2), r2=round(s.cmd_b, 2), q=round(s.q, 1), mass=round(s.mass, 1), thr=round(s.thrust_n),
                    gas=round(s.gas / V.RCS_GAS, 3), wd=round(self.wind.at(s.t, s.y), 2), on=int(s.on), vx=round(s.vx, 2), vy=round(s.vy, 2), om=round(s.om, 4),
                    ax=round(s.acc_x, 2), ay=round(s.acc_y, 2), dr=round(s.aero_f),
                    t={k: round(float(v), 2) for k, v in terms.items() if abs(v) > 1e-9})

    def _settle_verdict(self, new_td):
        """None while the lander is still settling after its first touch; 'rest' or 'crash' once it is decided."""
        s, td = self.s, self.td
        if new_td is not None:                                           # the instant of first contact
            if (td["vspeed_mps"] > C.SURVIVE_MAX_VSPEED_MPS or td["hspeed_mps"] > C.SURVIVE_MAX_HSPEED_MPS or td["tilt_rad"] > C.CONTACT_TILT_MAX):
                return "crash"
            if C.SETTLE_S <= 0:                                          # the old rule: judged at the instant of contact
                return "rest" if td["tilt_rad"] <= C.SURVIVE_MAX_TILT_RAD else "crash"
        tilt = abs(tilt_from_vertical(s.th))
        if tilt > C.TIP_OVER_RAD:                                        # it fell over
            self._finish_td()
            return "crash"
        speed = math.hypot(s.vx, s.vy)
        calm = tilt <= C.SUCCESS_MAX_TILT_RAD and speed <= C.SETTLE_REST_SPEED and abs(s.om) <= C.SETTLE_REST_SPIN
        self.settled_steps = self.settled_steps + 1 if calm else 0
        if self.settled_steps < int(round(C.SETTLE_HOLD_S * C.FPS)) and self.t < self.settle_end:
            return None
        self._finish_td()
        return "rest" if (tilt <= C.SURVIVE_MAX_TILT_RAD and speed <= 3.0 * C.SETTLE_REST_SPEED) else "crash"

    def _finish_td(self):
        """Where and how upright it ended up (the impact speed stays the speed of first contact)."""
        s, td = self.s, self.td
        td["dist_m"], td["tilt_rad"], td["settle_s"] = abs(s.x), abs(tilt_from_vertical(s.th)), (self.t - td["step"]) / C.FPS

    def _metrics(self, end: str, tilt: float, h: float) -> dict:
        s = self.s
        td = self.td
        oob = end == "oob"
        ok = (td is not None and td["dist_m"] <= C.SUCCESS_MAX_DIST_M and td["speed_mps"] <= C.SUCCESS_MAX_SPEED_MPS and td["tilt_rad"] <= C.SUCCESS_MAX_TILT_RAD
              and not s.out_of_prop and end == "rest")
        fuel_left = s.prop / self.budget if self.budget < 1e8 else 1.0
        m = dict(level=self.level, rules_stage=self.rules_stage, success=bool(ok), soft=M.is_soft_landing(td, end),
                 merit=M.merit(td, end, fuel_left, int(s.ign)), touchdown=td is not None,
                 dist_m=td["dist_m"] if td else abs(s.x), speed_mps=td["speed_mps"] if td else float("nan"),
                 tilt_rad=td["tilt_rad"] if td else abs(float(s.th)), fuel_used=(self.budget - s.prop) if self.budget < 1e8 else float("nan"),
                 out_of_fuel=bool(s.out_of_prop), end=end, crashed=end in ("crash", "oob", "tilt", "stall"), tilt_out=end == "tilt", stall_out=end == "stall",
                 ignitions=int(s.ign), steps=self.t, env_ok=self.env_ok_steps / self.air_steps if self.air_steps else float("nan"),
                 brake_peak=min(self.brake_peak, 3.0), touchdown_step=td["step"] if td else self.t)
        for i in range(len(self.band_n)):
            m[f"ov_{i}"] = self.band_sum[i] / self.band_n[i] if self.band_n[i] else float("nan")
        for k, v in self.terms_sum.items():
            m[f"term_{k}"] = v
        return m
