"""2-D rigid-body physics of the booster (replaces Box2D). Pure Python floats: fast for one vehicle, no dependencies.

Frame: x to the right (pad centre x = 0), y up (ground y = 0). The body axis e_b points from the engines to the nose:
e_b = (-sin th, cos th), e_n = (cos th, sin th) is the right-hand normal; th = 0 is upright, th > 0 tilts the nose to the left.
The vehicle is integrated about its centre of gravity (CG).

Forces: gravity, main-engine thrust (throttle, gimbal, altitude-dependent, mass flow), cold-gas RCS (two systems), grid-fin lift,
aerodynamic drag along the axis and cross-flow drag across it (with a centre of pressure above the CG, so engine-first flight is stable).
"""
from __future__ import annotations

import math

from src.sim import vehicle as V
from src.utils import config as C

G = C.GRAVITY_MPS2


class State:
    __slots__ = ("x", "y", "vx", "vy", "th", "om", "prop", "gas", "thr", "gim", "fin", "on", "ign", "t", "out_of_prop", "q", "thrust_n",
                 "cmd_a", "cmd_b", "acc_x", "acc_y", "aero_f", "contact", "pen_a", "pen_b")

    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0, th=0.0, om=0.0, prop=0.0):
        self.x, self.y, self.vx, self.vy, self.th, self.om = x, y, vx, vy, th, om
        self.prop = prop                  # kg of landing propellant on board
        self.gas = V.RCS_GAS              # kg of RCS nitrogen
        self.thr = 0.0                    # actual thrust as a fraction of full throttle (lags the command)
        self.gim = 0.0                    # actual gimbal angle (rad)
        self.fin = 0.0                    # actual fin deflection (rad)
        self.on = False                   # engine commanded running
        self.ign = 0                      # number of ignitions so far
        self.t = 0.0
        self.out_of_prop = False
        self.q = 0.0                      # dynamic pressure (Pa) of the last step
        self.thrust_n = 0.0
        self.cmd_a = 0.0                  # RCS commands actually applied (zero when the gas is gone / inside the deadband)
        self.cmd_b = 0.0
        self.acc_x = 0.0                  # proper acceleration (all forces except gravity) / mass, m/s^2, of the last step
        self.acc_y = 0.0
        self.aero_f = 0.0                 # magnitude of the aerodynamic force (N) of the last step
        self.contact = 0                  # feet in ground contact (0-2)
        self.pen_a = 0.0                  # how far each leg is compressed (m): the foot is this far below the ground before the leg gives
        self.pen_b = 0.0

    def copy(self) -> "State":
        s = State()
        for k in State.__slots__:
            setattr(s, k, getattr(self, k))
        return s

    @property
    def mass(self) -> float:
        return V.DRY_MASS + self.prop + self.gas

    def axis(self):
        return -math.sin(self.th), math.cos(self.th)

    def feet(self):
        """The two foot positions (x, y)."""
        ebx, eby = self.axis()
        enx, eny = math.cos(self.th), math.sin(self.th)
        bx, by = self.x - V.CG_BASE * ebx, self.y - V.CG_BASE * eby       # the engine bells
        half = V.LEG_SPAN / 2.0
        fx, fy = bx - V.FOOT_BELOW_BASE * ebx, by - V.FOOT_BELOW_BASE * eby
        return (fx - half * enx, fy - half * eny), (fx + half * enx, fy + half * eny)

    def foot_height(self) -> float:
        (_, ya), (_, yb) = self.feet()
        return min(ya, yb)


def step(s: State, throttle: float, gimbal: float, fin: float, rcs_a: float, rcs_b: float, dt: float, wind_x: float = 0.0) -> None:
    """Advance the state by dt seconds. throttle: commanded thrust fraction (0 = engine off, otherwise THROTTLE_MIN..1);
    gimbal / fin: commanded angles (rad); rcs_a / rcs_b: -1..1. Updates s in place."""
    m = s.mass
    sin_t, cos_t = math.sin(s.th), math.cos(s.th)
    ebx, eby = -sin_t, cos_t
    enx, eny = cos_t, sin_t
    h = max(s.y, 0.0)

    # ---- actuators (first-order lag with a rate limit)
    if throttle > 0.0 and not s.out_of_prop:
        if not s.on:
            s.ign += 1
        s.on = True
        target = min(1.0, max(V.THROTTLE_MIN, throttle))
    else:
        s.on = False
        target = 0.0
    tau = V.SPOOL_TAU if target > s.thr else V.SHUTDOWN_TAU
    s.thr += (target - s.thr) * (1.0 - math.exp(-dt / tau))
    gcmd = max(-V.GIMBAL_MAX, min(V.GIMBAL_MAX, gimbal))
    s.gim += max(-V.GIMBAL_RATE * dt, min(V.GIMBAL_RATE * dt, (gcmd - s.gim) * (1.0 - math.exp(-dt / V.GIMBAL_TAU))))
    fcmd = max(-V.FIN_MAX, min(V.FIN_MAX, fin))
    s.fin += max(-V.FIN_RATE * dt, min(V.FIN_RATE * dt, (fcmd - s.fin) * (1.0 - math.exp(-dt / V.FIN_TAU))))

    # ---- main engine
    fx = fy = torque = 0.0
    thrust = s.thr * V.thrust_full(h) if s.prop > 0.0 else 0.0
    s.thrust_n = thrust
    if thrust > 0.0:
        cg, sg = math.cos(s.gim), math.sin(s.gim)
        tx, ty = thrust * (cg * ebx + sg * enx), thrust * (cg * eby + sg * eny)
        fx += tx
        fy += ty
        rx, ry = -V.CG_BASE * ebx, -V.CG_BASE * eby           # the engine bells relative to the CG
        torque += rx * ty - ry * tx
        burned = s.thr * V.MDOT_FULL * dt
        s.prop -= burned
        if s.prop <= 0.0:
            s.prop = 0.0
            s.out_of_prop = True
            s.on = False
            s.thr = 0.0

    # ---- RCS (two lateral systems; each pushes along e_n at its own height)
    ca = rcs_a if abs(rcs_a) > V.RCS_DEADBAND and s.gas > 0.0 else 0.0
    cb = rcs_b if abs(rcs_b) > V.RCS_DEADBAND and s.gas > 0.0 else 0.0
    s.cmd_a, s.cmd_b = ca, cb
    if ca or cb:
        fa, fb = ca * V.RCS_FORCE, cb * V.RCS_FORCE
        fx += (fa + fb) * enx
        fy += (fa + fb) * eny
        torque += -(V.RCS_ARM_A * fa + V.RCS_ARM_B * fb)     # force along e_n at height arm above the CG: torque = -arm x force
        s.gas = max(0.0, s.gas - (abs(fa) + abs(fb)) / (V.RCS_ISP * V.G0) * dt)

    # ---- aerodynamics
    rho = V.density(h)
    vrx, vry = s.vx - wind_x, s.vy
    v_ax = vrx * ebx + vry * eby
    q_ax = 0.5 * rho * v_ax * v_ax
    s.q = 0.5 * rho * (vrx * vrx + vry * vry)
    cd = V.CD_AXIAL_BASE_FIRST if v_ax < 0.0 else V.CD_AXIAL_NOSE_FIRST
    d_ax = -q_ax * cd * V.BASE_AREA * (1.0 if v_ax >= 0.0 else -1.0)     # force along e_b opposing the axial motion
    fx += d_ax * ebx
    fy += d_ax * eby
    v_lat = vrx * enx + vry * eny
    for arm in V.CROSS_STATIONS:                                         # cross-flow drag on three slices of the body: net load above the CG (stable) and a good rotational damping
        v_n = v_lat - s.om * arm                                         # cross-flow speed at that slice (the rotation adds omega x arm)
        f_n = -0.5 * rho * V.CD_CROSS * V.SIDE_AREA / len(V.CROSS_STATIONS) * v_n * abs(v_n)
        fx += f_n * enx
        fy += f_n * eny
        torque += -arm * f_n
    f_fin = q_ax * V.FIN_AREA * V.FIN_CL_ALPHA * s.fin                   # grid fins: lift grows with the dynamic pressure along the axis
    fx += f_fin * enx
    fy += f_fin * eny
    torque += -V.FIN_ARM * f_fin

    # ---- the legs on the ground: each foot is a spring + damper (stroke limited) with sliding friction, acting at the foot
    s.contact, s.pen_a, s.pen_b = 0, 0.0, 0.0
    feet = s.feet()
    if min(feet[0][1], feet[1][1]) < 0.0:
        for i, (xf, yf) in enumerate(feet):
            pen = -yf
            if pen <= 0.0:
                continue
            rx, ry = xf - s.x, yf - s.y
            vfx, vfy = s.vx - s.om * ry, s.vy + s.om * rx                  # velocity of the foot point
            n = V.LEG_STIFFNESS * min(pen, V.LEG_STROKE) + V.LEG_BOTTOM_STIFFNESS * max(0.0, pen - V.LEG_STROKE) - V.LEG_DAMPING * vfy
            n = max(0.0, n)                                                # a leg only pushes
            ft = -V.FOOT_FRICTION * n * math.tanh(vfx / 0.1)
            fx += ft
            fy += n
            torque += rx * n - ry * ft
            s.contact += 1
            if i == 0:
                s.pen_a = min(pen, V.LEG_STROKE)
            else:
                s.pen_b = min(pen, V.LEG_STROKE)

    # ---- integrate (semi-implicit Euler)
    m = s.mass
    s.acc_x, s.acc_y = fx / m, fy / m
    s.aero_f = math.hypot(d_ax * ebx + f_fin * enx + f_n * enx, d_ax * eby + f_fin * eny + f_n * eny)
    s.vx += fx / m * dt
    s.vy += (fy / m - G) * dt
    s.om += torque / V.inertia(m) * dt
    s.x += s.vx * dt
    s.y += s.vy * dt
    s.th += s.om * dt
    s.t += dt


def passive_descent(s0: State, wind_x: float = 0.0, dt: float = 0.05, t_max: float = 120.0) -> tuple:
    """Where and how fast the vehicle reaches the ground if every actuator stays idle (drag included, attitude free).
    Returns (impact_x, impact_speed, time). The input state is not modified."""
    s = s0.copy()
    s.on, s.thr = False, 0.0
    while s.foot_height() > 0.0 and s.t - s0.t < t_max:
        step(s, 0.0, 0.0, 0.0, 0.0, 0.0, dt, wind_x)
    return s.x, math.hypot(s.vx, s.vy), s.t - s0.t


def powered_descent(s0: State, wind_x: float = 0.0, dt: float = 0.05, a_fraction: float = 0.8, v_touch: float = 1.6, t_max: float = 120.0) -> tuple:
    """Where the vehicle can come to rest over the ground with a normal landing: coast with drag until the braking profile says "burn now",
    then slow to v_touch over the remaining height while the sideways speed is also braked to zero at a constant rate (which covers half of
    vx x burn time). That is the pad position this start is "aimed" at; the *passive* path would land far from it, because the burn lasts
    ~9 s more than the fall. Returns (stop_x, burn_height, time)."""
    s = s0.copy()
    s.on, s.thr = False, 0.0
    while s.foot_height() > 0.0 and s.t - s0.t < t_max:
        h = max(s.foot_height(), 0.0)
        vd = max(-s.vy, 0.0)
        v = math.hypot(s.vx, s.vy)
        a_drag = 0.5 * V.density(h) * V.CD_AXIAL_BASE_FIRST * V.BASE_AREA * v * v / s.mass
        a_plan = max(a_fraction * (V.thrust_full(h) / s.mass - G), 2.0)
        if vd >= math.sqrt(2.0 * (a_plan + 0.4 * a_drag) * h + v_touch * v_touch) + 0.5 * V.SPOOL_TAU * a_plan:
            burn = 2.0 * h / (vd + v_touch)
            return s.x + 0.5 * s.vx * burn, h, s.t - s0.t + burn
        step(s, 0.0, 0.0, 0.0, 0.0, 0.0, dt, wind_x)
    return s.x, 0.0, s.t - s0.t


def min_landing_time(height_m: float, down_speed_mps: float, brake_mps2: float, g: float = G, land_speed_mps: float = 1.5) -> float:
    """Seconds of the fastest possible soft landing: fall freely (gravity only), then brake at brake_mps2 (net) so that the speed reaches
    land_speed_mps at the ground. The switch time is where distance fallen + braking distance = the height (bisection)."""
    v0, h = max(down_speed_mps, 0.0), max(height_m, 0.0)
    if h == 0.0:
        return 0.0

    def over(t):
        v = v0 + g * t
        return v0 * t + 0.5 * g * t * t + v * v / (2.0 * brake_mps2) - h

    ts = 0.0
    if over(0.0) < 0.0:
        lo, hi = 0.0, 1.0
        while over(hi) < 0.0:
            hi *= 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if over(mid) < 0.0 else (lo, mid)
        ts = 0.5 * (lo + hi)
    return ts + max(v0 + g * ts - land_speed_mps, 0.0) / brake_mps2
