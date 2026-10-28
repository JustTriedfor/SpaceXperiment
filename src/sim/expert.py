"""A scripted landing controller ("expert"): an engine-first descent, a throttle that tracks a braking profile, attitude control with
the right actuator for the airflow (fins at high dynamic pressure, gimbal under thrust, RCS as the backup), and gentle lateral steering.

It exists for three jobs: calibrating the fuel budget (how much propellant a good landing really needs), checking that the vehicle
numbers can land at all, and producing expert data for behaviour cloning. It is not meant to be optimal.
"""
from __future__ import annotations

import math

from src.sim import vehicle as V
from src.sim.physics import State, step
from src.utils import config as C

G = C.GRAVITY_MPS2

# tunables (fractions of the available authority)
A_PLAN_FRACTION = 0.6364        # the braking profile asks for this share of the vertical deceleration the engine could give at full throttle
V_TOUCH = 1.775                 # m/s the profile aims at over the ground
LATERAL_TILT_MAX = 0.7       # rad: the most the lateral steering tilts the thrust (high up)
LATERAL_TILT_GROUND = 0.16518    # rad: ... and the most it tilts it right at the ground
LATERAL_TILT_PER_M = 0.0095481   # rad per metre of height in between
AERO_HOLD_FRACTION = 0.7
AERO_FEEDFORWARD = False
AOA_CLIP = None
COAST_STEER = False
COAST_TC_MIN = 5.0
COAST_TC_FRAC = 0.5
COAST_AOA_MAX = 0.25              # rad
ATT_WN_MAX = 0.79443
ATT_ERR_REF = 0.35
ATT_ZETA = 1.0
CLIMB_CUT_MPS = 0.3
LATERAL_LEAD_S = 1.2756
LATERAL_MODE = "poly"
LATERAL_KE = 4.0
LATERAL_KX = 0.03
LATERAL_KV = 0.6
LATERAL_V_MAX = 30.0
LATERAL_A_BRAKE = 2.0
LATERAL_DEAD_M = 1.0
LATERAL_TAU_V = 2.5
LATERAL_VX_GAIN = 3.6486
LATERAL_X_FADE_M = 269.94
LATERAL_TGO_SCALE = 0.38003
LATERAL_TGO_MIN = 3.4092
LATERAL_TC_UNUSED = 4              # s: time constant of the lateral pull
ALIGN_RAMP_M = 2500          # the nose swings from "into the airflow" to vertical over the last this-many metres before the burn


def _drag_decel(s: State, h: float) -> float:
    """Drag deceleration (m/s^2) along the descent at this speed, engine-first."""
    v = math.hypot(s.vx, s.vy)
    return 0.5 * V.density(h) * V.CD_AXIAL_BASE_FIRST * V.BASE_AREA * v * v / s.mass


def _aero_torque(s: State, vrx: float, vry: float, th: float, h: float) -> float:
    """Torque (N m, about the CG) of the cross-flow drag if the stage were at attitude th (same slices as the simulator)."""
    rho = V.density(h)
    enx, eny = math.cos(th), math.sin(th)
    v_lat = vrx * enx + vry * eny
    tq = 0.0
    for arm in V.CROSS_STATIONS:
        v_n = v_lat - s.om * arm
        f_n = -0.5 * rho * V.CD_CROSS * V.SIDE_AREA / len(V.CROSS_STATIONS) * v_n * abs(v_n)
        tq += -arm * f_n
    return tq


class Expert:
    def __init__(self, a_plan_fraction: float = A_PLAN_FRACTION):
        self.frac = a_plan_fraction
        self.burning = False

    def reset(self):
        self.burning = False

    def act(self, s: State, wind_x: float = 0.0):
        h = max(s.foot_height(), 0.0)
        m = s.mass
        vd = max(-s.vy, 0.0)
        t_full = V.thrust_full(h)
        cos_t = max(math.cos(s.th), 0.3)
        a_max = t_full / m - G                                   # vertical deceleration at full throttle if the nose were vertical (no drag)
        a_plan = max(self.frac * a_max, 2.0)
        a_drag = _drag_decel(s, h)
        # braking profile: the speed from which a_plan (plus part of the drag) still stops us at V_TOUCH over the ground
        v_des = math.sqrt(2.0 * (a_plan + 0.4 * a_drag) * h + V_TOUCH * V_TOUCH)
        if not self.burning and vd >= v_des + 0.5 * V.SPOOL_TAU * a_plan and s.prop > 0.0:
            self.burning = True
        if s.prop <= 0.0:
            self.burning = False
        if self.burning and s.vy > CLIMB_CUT_MPS and s.thr > 0.0:        # stopped above the ground and started to climb (min throttle cannot hover): cut, fall, burn again
            self.burning = False

        # ---- throttle: brake at exactly the deceleration the remaining distance asks for (re-evaluated every step, so lag and mass loss are absorbed)
        throttle = 0.0
        if self.burning:
            a_req = max((vd * vd - V_TOUCH * V_TOUCH) / (2.0 * max(h, 0.3)), 0.0) - 0.5 * a_drag
            u = m * (a_req + G) / (t_full * cos_t)
            throttle = max(V.THROTTLE_MIN, min(1.0, u))

        # ---- the attitude we want
        vrx, vry = s.vx - wind_x, s.vy
        flow = math.atan2(vrx, -vry) if (vrx * vrx + vry * vry) > 25.0 else 0.0         # nose against the airflow
        if self.burning:
            # lateral guidance: reach x = 0 with vx = 0 when the descent ends (polynomial / zero-effort-miss law on the time left)
            t_go = max(LATERAL_TGO_SCALE * 2.0 * h / (vd + V_TOUCH), LATERAL_TGO_MIN)
            x_l = s.x + s.vx * LATERAL_LEAD_S + 0.5 * s.acc_x * LATERAL_LEAD_S ** 2          # look ahead by the lag of the tilt loop
            vx_l = s.vx + s.acc_x * LATERAL_LEAD_S
            x_w = min(1.0, h / LATERAL_X_FADE_M)                                            # near the ground stop drifting; do not chase the last metres of position
            if LATERAL_MODE == "stop":
                e = x_w * (s.x + 0.5 * s.vx * t_go)                                          # where a constant-rate sideways stop would end, relative to the pad
                a_lat = (s.vx + LATERAL_KE * e / t_go) / t_go
            elif LATERAL_MODE == "pd":
                a_lat = LATERAL_KX * x_w * s.x + LATERAL_KV * s.vx
            else:
                a_lat = 6.0 * x_w * x_l / (t_go * t_go) + LATERAL_VX_GAIN * vx_l / t_go
            lat = a_lat * m / max(t_full * max(s.thr, 0.4), 1.0)
            tilt_cap = min(LATERAL_TILT_MAX, LATERAL_TILT_GROUND + LATERAL_TILT_PER_M * h)
            th_d = max(-tilt_cap, min(tilt_cap, lat))
        else:
            b = max(0.0, min(1.0, 1.0 - (h - (vd * vd / (2.0 * a_plan))) / ALIGN_RAMP_M))      # how close the burn is
            th_d = (1.0 - b) * flow
            # steer sideways with the angle of attack: the body's cross-flow force pushes the stage toward the pad (fins hold the angle)
            vspeed = math.hypot(vrx, vry)
            if COAST_STEER and vspeed > 60.0 and h > 150.0:
                t_go = max((h + vd * vd / (2.0 * a_plan)) / max(vd, 30.0), 3.0)
                a_des = -(6.0 * s.x / (t_go * t_go) + 4.0 * s.vx / t_go)                  # polynomial guidance: wanted lateral acceleration (m/s^2, +x)
                q_cross = 0.5 * V.density(h) * V.CD_CROSS * V.SIDE_AREA * vspeed * vspeed / m
                sin2 = min(abs(a_des) / max(q_cross, 1e-6), math.sin(COAST_AOA_MAX) ** 2)
                th_d += math.copysign(math.asin(math.sqrt(sin2)), a_des) * (1.0 - b)
        # ---- the actuators must be able to hold the angle of attack this attitude implies: shrink the tilt away from the airflow until they can
        q_ax = 0.5 * V.density(h) * ((vrx * -math.sin(s.th) + vry * math.cos(s.th)) ** 2)
        fin_gain = V.FIN_ARM * q_ax * V.FIN_AREA * V.FIN_CL_ALPHA                       # N m per rad of fin
        gim_gain = V.CG_BASE * max(s.thrust_n, 0.0)                                      # N m per rad of gimbal
        auth = AERO_HOLD_FRACTION * (fin_gain * V.FIN_MAX + gim_gain * V.GIMBAL_MAX + V.RCS_FORCE * V.RCS_ARM_A)
        base = flow if (vrx * vrx + vry * vry) > 25.0 else s.th
        if AERO_FEEDFORWARD and abs(_aero_torque(s, vrx, vry, th_d, h)) > auth:
            lo, hi = 0.0, 1.0
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                if abs(_aero_torque(s, vrx, vry, base + mid * (th_d - base), h)) > auth:
                    hi = mid
                else:
                    lo = mid
            th_d = base + lo * (th_d - base)
        if AOA_CLIP is not None and (vrx * vrx + vry * vry) > 400.0:
            th_d = max(flow - AOA_CLIP, min(flow + AOA_CLIP, th_d))                          # the airflow holds the nose: do not ask for more angle of attack than the fins can keep
        self.dbg = (th_d, s.th, s.om)
        # ---- attitude control: angular acceleration command, then the actuator with authority
        I = V.inertia(m)
        alpha_avail = max(fin_gain * V.FIN_MAX / I, gim_gain * V.GIMBAL_MAX / I, V.RCS_FORCE * V.RCS_ARM_A / I)
        wn = max(0.15, min(ATT_WN_MAX, math.sqrt(alpha_avail / ATT_ERR_REF)))
        alpha_cmd = wn * wn * (th_d - s.th) - 2.0 * ATT_ZETA * wn * s.om
        torque = I * alpha_cmd - (_aero_torque(s, vrx, vry, s.th, h) if AERO_FEEDFORWARD else 0.0)                      # the actuators supply what the airflow does not
        fin = gimbal = rcs_a = rcs_b = 0.0
        if fin_gain > 1.5e5:
            fin = max(-V.FIN_MAX, min(V.FIN_MAX, -torque / fin_gain))
            torque_left = torque + fin_gain * fin
        else:
            torque_left = torque
        if gim_gain > 2e5:
            gimbal = max(-V.GIMBAL_MAX, min(V.GIMBAL_MAX, torque_left / gim_gain))
            torque_left -= gim_gain * gimbal
        if abs(torque_left) > 0.0 and fin_gain < 1.5e5:
            rcs_a = max(-1.0, min(1.0, -torque_left / (V.RCS_ARM_A * V.RCS_FORCE)))
        return throttle, gimbal, fin, rcs_a, rcs_b


def rollout(s0: State, wind=None, expert: Expert | None = None, dt: float = 0.01, control_dt: float = 0.05, t_max: float = 120.0) -> dict:
    """Fly the expert from s0 to touchdown. Returns the outcome."""
    s = s0.copy()
    ex = expert or Expert()
    ex.reset()
    cmd = (0.0, 0.0, 0.0, 0.0, 0.0)
    next_ctl = 0.0
    peak_q = 0.0
    prev_v = (s.vx, s.vy)
    while s.t < t_max:
        w = wind.at(s.t, s.y) if wind else 0.0
        if s.t >= next_ctl:
            cmd = ex.act(s, w)
            next_ctl += control_dt
        prev_v = (s.vx, s.vy)
        step(s, *cmd, dt, w)
        peak_q = max(peak_q, s.q)
        if s.foot_height() <= 0.0:
            break
    speed = math.hypot(*prev_v)
    return dict(t=s.t, x=s.x, vx=prev_v[0], vy=prev_v[1], speed=speed, tilt=abs(s.th), prop_used=s0.prop - s.prop, prop_left=s.prop, ignitions=s.ign,
                out_of_prop=s.out_of_prop, peak_q=peak_q, landed=s.foot_height() <= 0.0, gas_used=V.RCS_GAS - s.gas)
