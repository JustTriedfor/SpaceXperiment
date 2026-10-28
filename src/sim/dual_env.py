"""DualEnv ("Dual land"): ONE environment with TWO landers and TWO pads.

Both landers start together DUAL_SPAWN_HEIGHT_M up on the same predetermined path: one start is drawn per episode (speed, entry angle, side, wind)
and both landers get it, in their own pad-relative coordinates, plus a small independent jitter (DUAL_JITTER). Their pads are DUAL_PAD_SPACING_M
apart; the landers start that far apart too (the display puts lander A over the left pad and B over the right one; the physics and each lander's
observation are pad-relative, so the single-lander skills carry over unchanged).

One policy steers both: observation = [A's 18, B's 18, height gap, fall-speed gap, A down, B down] (40), action = [A's 5, B's 5] (10).
A lander that is down stays where it is (its actions are ignored) until the other one is down too: only then does the next episode start.
The goal is to land together: reward = both landers' rewards + a potential-shaping term for closing the height gap while both are in the air
+ a bonus when both survive and touch down within DUAL_SYNC_TOL_S of each other.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.sim.env import OBS_DIM as OBS_ONE, SpaceXEnv
from src.sim.spawn import Wind, make_spawn
from src.training import reward as R
from src.utils import config as C

END_RANK = {"rest": 0, "timeout": 1, "stall": 2, "tilt": 3, "oob": 4, "crash": 5}      # which end to report for the pair: the worst one


def _jitter(s, rng, j: float):
    """A small independent difference to one lander's start (j = 0 leaves it exactly on the shared path)."""
    if j <= 0.0:
        return s
    f = 1.0 + 0.03 * j * float(rng.normal())
    s.vx *= f
    s.vy *= f
    s.th += math.radians(1.0) * j * float(rng.normal())
    s.om += 0.01 * j * float(rng.normal())
    s.x += 8.0 * j * float(rng.normal())
    return s


def _shift(f: dict, dx: float) -> dict:
    g = dict(f)
    g["x"] = round(f["x"] + dx, 3)
    g["legs"] = [[round(l[0] + dx, 3), l[1], l[2], l[3]] for l in f["legs"]]
    return g


def _idle(f: dict) -> dict:
    """The frame of a lander that is already down: it stays put, engines off."""
    g = dict(f)
    g.update(m=0.0, s=0.0, r1=0.0, r2=0.0, on=0, thr=0, vx=0.0, vy=0.0, om=0.0, ax=0.0, ay=0.0, q=0.0, dr=0, t={})
    return g


def agent_view(joint, k: int):
    """One lander's view of the dual observation, for the two-network setup (each network sees its OWN lander first):
    [own 18, partner 18, height gap (own - partner) / 100, fall-speed gap / 50, own down, partner down]. Works on (40,) or (n, 40) arrays."""
    j = np.asarray(joint)
    a, b = j[..., :OBS_ONE], j[..., OBS_ONE:2 * OBS_ONE]
    gap, dvy, da, db = j[..., 2 * OBS_ONE], j[..., 2 * OBS_ONE + 1], j[..., 2 * OBS_ONE + 2], j[..., 2 * OBS_ONE + 3]
    if k == 0:
        return np.concatenate([a, b, np.stack([gap, dvy, da, db], axis=-1)], axis=-1).astype(np.float32)
    return np.concatenate([b, a, np.stack([-gap, -dvy, db, da], axis=-1)], axis=-1).astype(np.float32)


class DualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, level: int = 0, fuel_budget: float | None = None, record: bool = False, randomize: bool = True):
        super().__init__()
        self.level = int(level)
        self.record = record
        self.a = SpaceXEnv(level=level, fuel_budget=fuel_budget, record=record)
        self.b = SpaceXEnv(level=level, fuel_budget=fuel_budget, record=record)
        self.observation_space = spaces.Box(-20.0, 20.0, (C.DUAL_OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (C.DUAL_ACT_DIM,), dtype=np.float32)
        self.rules_stage = self.a.rules_stage
        self._reset_state()

    # ------------------------------------------------------------------ curriculum hooks (called through env_method)
    def set_level(self, level: int) -> None:
        self.level = int(level)
        self.a.set_level(level)
        self.b.set_level(level)

    def set_rules_stage(self, stage: int) -> None:
        self.a.set_rules_stage(stage)
        self.b.set_rules_stage(stage)

    def _reset_state(self):
        self.t = 0
        self.done = [False, False]
        self.end = [None, None]
        self.t_end = [0, 0]
        self.metrics = [None, None]
        self.infos = [None, None]
        self.obs = [np.zeros(OBS_ONE, np.float32), np.zeros(OBS_ONE, np.float32)]
        self.gap_prev = 0.0
        self.sync_steps: list = []
        self.sync_terminal = 0.0
        self.spacing = float(C.DUAL_PAD_SPACING_M)

    # ------------------------------------------------------------------ observation
    def _heights(self):
        return max(self.a.s.foot_height(), 0.0), max(self.b.s.foot_height(), 0.0)

    def _obs(self) -> np.ndarray:
        ha, hb = self._heights()
        extra = [float(np.clip((ha - hb) / 100.0, -20, 20)), float(np.clip((self.a.s.vy - self.b.s.vy) / 50.0, -20, 20)), float(self.done[0]), float(self.done[1])]
        return np.clip(np.concatenate([self.obs[0], self.obs[1], np.asarray(extra, np.float32)]), -20.0, 20.0).astype(np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        rng = self.np_random
        cfg = C.LEVELS[self.level]
        wind = Wind(cfg["wind_power"], rng)
        s0, d0 = make_spawn(rng, self.level, prop=self.a.budget, height_m=C.DUAL_SPAWN_HEIGHT_M, wind=wind)      # the shared predetermined path
        j = float(C.DUAL_JITTER)
        starts = [_jitter(s0.copy(), rng, j), _jitter(s0.copy(), rng, j)]
        for k, env in enumerate((self.a, self.b)):
            self.obs[k], _ = env.reset(seed=int(rng.integers(1 << 31)), options=dict(spawn=(starts[k], d0), wind=wind))
        self.rules_stage = self.a.rules_stage
        ha, hb = self._heights()
        self.gap_prev = abs(ha - hb)
        info = dict(level=self.level, rules_stage=self.rules_stage, spawn=dict(d0, spacing=self.spacing), time_limit_s=self.a.limit_s)
        return self._obs(), info

    def step(self, action):
        act = np.clip(np.asarray(action, dtype=np.float64).reshape(C.DUAL_ACT_DIM), -1.0, 1.0)
        total = 0.0
        own = [0.0, 0.0]                                                   # each lander's own reward (for the two-network setup)
        was_done = list(self.done)
        for k, env in enumerate((self.a, self.b)):
            if self.done[k]:
                continue
            o, r, term, trunc, info = env.step(act[5 * k:5 * k + 5])
            self.obs[k] = o
            total += r
            own[k] = r
            if term or trunc:
                self.done[k], self.end[k], self.t_end[k] = True, info["episode_metrics"]["end"], info["episode_metrics"]["touchdown_step"]
                self.metrics[k], self.infos[k] = info["episode_metrics"], info
        self.t += 1
        # ---- the landing-together terms
        shape = 0.0
        if not any(was_done) and not any(self.done):                      # both still in the air: reward closing the height gap (potential shaping)
            ha, hb = self._heights()
            gap = abs(ha - hb)
            shape = C.DUAL_SYNC_SHAPE * (self.gap_prev - gap)
            self.gap_prev = gap
        bonus = 0.0
        both_down = all(self.done)
        dt_s = abs(self.t_end[0] - self.t_end[1]) / C.FPS
        if both_down and self.end[0] == "rest" and self.end[1] == "rest":
            bonus = C.DUAL_SYNC_BONUS * max(0.0, 1.0 - dt_s / max(C.DUAL_SYNC_TOL_S, 1e-6))
        shared = (shape + bonus) * R.GLOBAL_SCALE                          # the landing-together terms are shared: both landers earn them in full
        total += shared
        self.sync_steps.append(shape)
        self.sync_terminal = bonus
        info = dict(time_s=self.t / C.FPS, down=list(self.done), agent_rewards=(own[0] + shared, own[1] + shared))
        terminated = truncated = False
        if both_down:
            truncated = any(e == "timeout" for e in self.end)
            terminated = not truncated
            info["episode_metrics"] = self._metrics(dt_s)
            if self.record:
                info["trajectory"] = self._trajectory(info["episode_metrics"], dt_s)
        return self._obs(), float(total), terminated, truncated, info

    # ------------------------------------------------------------------ outputs
    def _metrics(self, dt_s: float) -> dict:
        ma, mb = self.metrics
        both_rest = self.end[0] == "rest" and self.end[1] == "rest"
        sync_ok = both_rest and dt_s <= C.DUAL_SYNC_TOL_S
        nan_mean = lambda *v: float(np.nanmean(v)) if not all(x != x for x in v) else float("nan")
        nan_max = lambda *v: float(np.nanmax(v)) if not all(x != x for x in v) else float("nan")
        end = max(self.end, key=lambda e: END_RANK.get(e, 9))
        m = dict(level=ma["level"], rules_stage=ma["rules_stage"], success=bool(ma["success"] and mb["success"] and sync_ok),
                 soft=bool(ma["soft"] and mb["soft"] and sync_ok), merit=min(ma["merit"], mb["merit"]), touchdown=bool(ma["touchdown"] and mb["touchdown"]),
                 dist_m=nan_max(ma["dist_m"], mb["dist_m"]), speed_mps=nan_max(ma["speed_mps"], mb["speed_mps"]), tilt_rad=nan_max(ma["tilt_rad"], mb["tilt_rad"]),
                 fuel_used=ma["fuel_used"] + mb["fuel_used"], out_of_fuel=bool(ma["out_of_fuel"] or mb["out_of_fuel"]), end=end,
                 crashed=bool(ma["crashed"] or mb["crashed"]), tilt_out=bool(ma["tilt_out"] or mb["tilt_out"]), stall_out=bool(ma["stall_out"] or mb["stall_out"]),
                 ignitions=int(ma["ignitions"] + mb["ignitions"]), steps=max(ma["steps"], mb["steps"]), env_ok=nan_mean(ma["env_ok"], mb["env_ok"]),
                 brake_peak=nan_max(ma["brake_peak"], mb["brake_peak"]), sync_dt_s=dt_s if both_rest else float("nan"), sync_ok=bool(sync_ok))
        for k in ma:
            if k.startswith("ov_"):
                m[k] = nan_mean(ma[k], mb[k])
        keys = {k for k in list(ma) + list(mb) if k.startswith("term_")}
        for k in keys:
            m[k] = ma.get(k, 0.0) + mb.get(k, 0.0)
        m["term_sync_shape"] = float(sum(self.sync_steps))
        m["term_sync_bonus"] = float(self.sync_terminal)
        return m

    def _trajectory(self, m: dict, dt_s: float) -> dict:
        base = self.infos[0]["trajectory"]
        n = max(len(self.a.frames), len(self.b.frames))
        sp = self.spacing
        fa, fb = list(self.a.frames), list(self.b.frames)
        out = []
        for i in range(n):
            fia = fa[i] if i < len(fa) else _idle(fa[-1])
            fib = fb[i] if i < len(fb) else _idle(fb[-1])
            t = dict(fia.get("t", {}))                                   # lander A's own terms (+ the shared landing-together terms); B keeps its own
            if i < len(self.sync_steps) and abs(self.sync_steps[i]) > 1e-9:
                t["sync"] = round(self.sync_steps[i], 2)
            if i == n - 1 and self.sync_terminal:
                t["sync"] = round(t.get("sync", 0.0) + self.sync_terminal, 2)
            A, B = _shift(fia, -sp / 2), _shift(fib, +sp / 2)
            B["t"] = dict(fib.get("t", {}))
            A["t"] = t
            A["b"] = B
            out.append(A)
        ra, rb = self.infos[0]["trajectory"]["result"], self.infos[1]["trajectory"]["result"]
        both = self.end[0] == "rest" and self.end[1] == "rest"
        score = sum(sum(f["t"].values()) + sum(f["b"]["t"].values()) for f in out)
        res = dict(end=m["end"], success=bool(m["success"]), soft=bool(m["soft"]), merit=round(float(m["merit"]), 1),
                   speed=None if m["speed_mps"] != m["speed_mps"] else round(float(m["speed_mps"]), 3), dist=round(float(m["dist_m"]), 3),
                   fuel_out=bool(m["out_of_fuel"]), score=round(score, 1))
        traj = dict(base, frames=out, result=res)
        traj["pad"] = [-C.PAD_HALF_WIDTH_M - sp / 2, C.PAD_HALF_WIDTH_M - sp / 2, 0.0]
        traj["dual"] = dict(spacing=sp, pads=[-sp / 2, sp / 2], tol_s=C.DUAL_SYNC_TOL_S, sync_dt=round(dt_s, 2) if both else None, sync_ok=bool(m["sync_ok"]),
                            results=[ra, rb], ends=list(self.end), td_steps=list(self.t_end))
        return traj
