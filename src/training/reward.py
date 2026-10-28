"""Custom reward v2: potential-based shaping + survive-gated landing score.  (Replaces the plan's table.)

Why the old table failed (measured, see README): its positive landing bonuses were paid on touchdown even for
crashes (72% of crashes scored > 0), "drift" paid gravity for falling, and nothing told the agent WHEN to
brake. A touchdown above ~2 m/s is fatal (<=1.5 m/s survives 98%, >=3 m/s never), so from a high spawn the
agent never saw a survivable landing and learned nothing.

Design
  1. Dense guidance = potential-based shaping, F = phi(s') - phi(s). A potential cannot be farmed (flying a
     loop returns to the same phi), unlike a per-step bonus. phi(s) is how good the state is:
        - distance to the pad centre                          -W_DIST  per m
        - speed ABOVE the safe-descent envelope               -W_VEL   per m/s of excess
        - tilt and spin                                       -W_ATT per rad, -W_SPIN per rad/s
        - legs on the ground                                  +W_LEG   per leg
     The envelope is the speed from which the lander can still stop: v_env(h) = sqrt(2 a h) + v_floor, with
     a = ENVELOPE_DECEL (65% of the ~8.9 m/s^2 the lander can deliver, measured) and v_floor = 1.5 m/s (the
     measured safe touchdown speed). Falling faster than the envelope lowers phi, braking raises it, so the
     agent gets paid at the moment it should burn. A free fall nets negative (distance gain < velocity loss).
  2. Time and fuel are small per-step costs (hovering can never pay).
  3. The landing score is paid only when the episode ends with the lander AT REST (a survived landing).
     A crash, out-of-bounds or timeout earns CRASH_PENALTY and none of it.
Distances in m, speeds in m/s, angles in rad. Returned values are unscaled; CustomReward multiplies by GLOBAL_SCALE.
"""
from __future__ import annotations

import math
from typing import NamedTuple

from src.utils import config as C

# ---- potential weights (points)
W_DIST = 0.5  # per m from the pad centre (spawn is ~3 km away)
W_VEL = 10.0  # per m/s above the safe-descent envelope
W_ATT = 100.0         # per rad of tilt
W_SPIN = 100.0  # per rad/s of angular velocity (real spins are tenths of a rad/s)
W_LEG = 20.0          # per leg in contact
# ---- descent envelope (calibrated from measurements, see module docstring)
ENVELOPE_DECEL = 14.0  # m/s^2 planned net braking (about 65% of the ~22 m/s^2 the engine can give at full throttle, see src/sim/calibrate.py)
ENVELOPE_FLOOR = 3.0  # m/s allowed at touchdown (the success limit)
# ---- per-step costs
TIME_PENALTY = -0.05
FUEL_PER_UNIT = -0.05
# ---- slow-approach charge: scales EXPONENTIALLY with altitude (agents bought time by braking early and
#      creeping down: a fixed "slower than 3 m/s" rule never fired because they crawled at 5-10 m/s)
SLOW_LIMIT_FRAC = 0.90    # the speed floor at height h is this fraction of the safe-descent envelope speed
SLOW_RATE_REF = 0.5  # points per SECOND at SLOW_REF_HEIGHT for a lander that is (nearly) stopped (first run: 4.0 made the charge -24,000 per episode and taught free fall)
SLOW_REF_HEIGHT = 100.0  # m
SLOW_GROWTH = 100.0       # the rate is multiplied by this every SLOW_GROWTH_SPAN metres of altitude
SLOW_GROWTH_SPAN = 300.0  # m   (the rate is multiplied by SLOW_GROWTH every this many metres)
SLOW_MIN_HEIGHT = 30.0  # m: the final approach below this is exempt
SLOW_MAX_HEIGHT = 400.0  # m: the exponential is capped here (about 50 points/s for a lander that is stopped)
ASCEND_SPEED = 5.0  # m/s of upward speed that counts as ascending
ASCEND_PENALTY = -1000.0  # paid once per episode
# ---- landing score (only for a survived landing), by zone. The green zone is the pad (radius 2 m).
LAND_OFF_PAD = 100.0      # survived, but outside the green zone
LAND_ZONE = 500.0         # survived inside the green zone, not a soft landing
LAND_SOFT_EDGE = 700.0    # soft landing in the zone with every parameter at its limit
LAND_PERFECT = 10000.0    # soft landing, bullseye, zero speed, perfectly upright
SOFT_SPEED = 3.0  # m/s   soft landing = within the zone, <= SOFT_SPEED and <= SOFT_TILT
SOFT_TILT = 0.15          # rad
ACC_W_DIST, ACC_W_SPEED, ACC_W_TILT = 0.35, 0.45, 0.20     # weights of the accuracy index
# ---- crash scoring = TWO INDEPENDENT scores added together, so the agent can tell when it is getting closer:
#   POSITION: where it hit.  Inside the green zone: +500 at the edge rising exponentially to +1000 at the centre.
#             Outside: -1000 just past the edge, growing exponentially per metre of miss (capped).
#   SPEED:    how hard it hit.  Goal 1.5 m/s; every m/s over the goal costs 100 points.
CRASH_IN_EDGE = 500.0        # crash at the zone edge (2 m from the centre)
CRASH_IN_CENTER = 1000.0     # crash dead centre
CRASH_OUT_EDGE = -1000.0     # crash just outside the zone
CRASH_OUT_GROWTH = 1.012  # the outside penalty is multiplied by this for every metre beyond the edge (reaches the cap at ~200 m)
CRASH_OUT_CAP = -10000.0     # the outside penalty never goes below this
CRASH_SPEED_GOAL = 3.0  # m/s
CRASH_SPEED_PER_MPS = -100.0  # points per m/s above the goal (the PPO run from the cloned start traded touchdown speed for accuracy at -25: 13 -> 20 m/s while the miss fell to 7 m)
CRASH_POS_FADE_MPS = 15.0  # a crash-position BONUS fades to 0 as the overspeed grows to this (60 was too lenient: a 20 m/s crash still kept 70% of it)
CRASH_PENALTY = -250.0       # independent flat penalty for every crash / out-of-bounds, on top of the position and speed scores
CRASH_VERTICAL_BONUS = 0.0   # optional extra for a crash inside the zone within VERTICAL_DEG of vertical (off by default)
VERTICAL_DEG = 5.0
# ---- off-course charge: where would the lander land if the engines cut out now?
OFFCOURSE_PER_SEC = -10.0    # points per second while that predicted touchdown is outside the tolerance
OFFCOURSE_TOLERANCE_M = 30.0  # m: predicted touchdown farther than this from the pad costs points
# ---- tilt-out: the lander was terminated for tilting more than TILT_LIMIT_DEG from vertical
TILT_OUT_PENALTY = -1000.0
# 1: tilt-out = -1000 PLUS the crash it was heading for (position and speed scored from its ballistic path, or from
#    its landing if it had already touched down). 0: a flat -1000. Flat is exploitable: dying early (-1000) would beat
#    a free fall (about -1850), so the lander could learn to throw itself over on purpose.
TILT_OUT_PROJECTED = 1.0
# ---- main-engine re-ignitions: a real Falcon 9 lights its landing engine once. The first ignition is free; every
#      re-ignition after it costs REFIRE_BASE x REFIRE_GROWTH^(n-2) (n = the ignition number), up to MAX_IGNITIONS.
FIRST_IGNITION_BONUS = 500.0   # paid once, when the main engine is lit for the first time
REFIRE_BASE = 25.0
REFIRE_GROWTH = 2.0
# ---- stall: airborne and (nearly) stopped before touchdown. Zero tolerance: terminated, -1500.
STALL_OUT_PENALTY = -1500.0
STALL_OUT_PROJECTED = 1.0    # same guard as the tilt-out: also score the crash it was heading for (0 = flat)
# ---- penalties
TIMEOUT_PENALTY = -1000.0   # the lander did not touch down within the time limit (a crash or leaving the bounds is scored by the two crash scores above)
FUEL_OUT_PENALTY = -150.0   # paid once when the fuel hits zero
GLOBAL_SCALE = 0.005          # landing scores reach 10,000, so the scale is lower than before

# Web-app overrides (see config.py); only names defined above can be overridden.
for _k, _v in C.OVERRIDES.get("reward", {}).items():
    if _k in globals() and _k.isupper():
        globals()[_k] = _v


class State(NamedTuple):
    dist: float      # m from the pad centre (at leg-contact height)
    speed: float     # m/s, |velocity|
    height: float    # m of the lander centre above the leg-contact level
    angle: float     # rad (not wrapped: spinning accumulates)
    spin: float      # rad/s
    legs: int        # legs in ground contact (0-2)
    vy: float = 0.0  # m/s vertical, positive = up
    x: float = 0.0   # m horizontal offset from the pad centre
    vx: float = 0.0  # m/s horizontal


def envelope_speed(height_m: float) -> float:
    """Fastest speed (m/s) from which the lander can still stop before the ground at this height."""
    return math.sqrt(2.0 * ENVELOPE_DECEL * max(height_m, 0.0)) + ENVELOPE_FLOOR


def potential_parts(s: State) -> dict:
    """phi(s), split into its parts (all <= 0 except legs)."""
    return dict(
        dist=-W_DIST * s.dist,
        vel=-W_VEL * max(0.0, s.speed - envelope_speed(s.height)),
        att=-W_ATT * abs(s.angle) - W_SPIN * abs(s.spin),
        legs=W_LEG * s.legs,
    )


def potential(s: State) -> float:
    return sum(potential_parts(s).values())


def landing_accuracy(touchdown: dict) -> float:
    """Accuracy index in [0, 1] of a soft landing in the zone: weighted geometric mean of how far inside each
    limit it is (distance to centre, touchdown speed, tilt). Any parameter at its limit gives 0; all perfect gives 1."""
    a_d = max(0.0, 1.0 - touchdown["dist_m"] / C.SUCCESS_MAX_DIST_M)
    a_v = max(0.0, 1.0 - touchdown["speed_mps"] / SOFT_SPEED)
    a_t = max(0.0, 1.0 - touchdown["tilt_rad"] / SOFT_TILT)
    return (a_d ** ACC_W_DIST) * (a_v ** ACC_W_SPEED) * (a_t ** ACC_W_TILT)


def landing_score(touchdown: dict) -> float:
    """Total landing points for a survived landing: 100 off the pad, 500 in the zone, and for a soft landing in the
    zone 700 at the limits rising EXPONENTIALLY with accuracy to 10,000 for a perfect bullseye."""
    in_zone = touchdown["dist_m"] <= C.SUCCESS_MAX_DIST_M
    soft = touchdown["speed_mps"] <= SOFT_SPEED and touchdown["tilt_rad"] <= SOFT_TILT
    if not in_zone:
        return LAND_OFF_PAD
    if not soft:
        return LAND_ZONE
    return LAND_SOFT_EDGE * (LAND_PERFECT / LAND_SOFT_EDGE) ** landing_accuracy(touchdown)


def landing_terms(touchdown: dict) -> dict:
    """Landing points split for logging: 'land' = the zone tier, 'soft' = the exponential part above 700."""
    total = landing_score(touchdown)
    base = min(total, LAND_SOFT_EDGE)
    return dict(land=base, soft=total - base)


def slow_rate_per_sec(s: State) -> float:
    """Points per second (>= 0) charged for being slower than the altitude-dependent speed floor.

    rate = R(h) x shortfall,  R(h) = SLOW_RATE_REF * SLOW_GROWTH ** ((h - SLOW_REF_HEIGHT) / SLOW_GROWTH_SPAN)
    shortfall = max(0, 1 - speed / (SLOW_LIMIT_FRAC * envelope_speed(h))).  Zero on the ground and below
    SLOW_MIN_HEIGHT. The floor sits under the envelope, so a corridor of unpunished speeds remains."""
    if s.legs > 0 or s.height <= SLOW_MIN_HEIGHT:
        return 0.0
    h = min(s.height, SLOW_MAX_HEIGHT)
    shortfall = max(0.0, 1.0 - s.speed / (SLOW_LIMIT_FRAC * envelope_speed(h)))
    return SLOW_RATE_REF * SLOW_GROWTH ** ((h - SLOW_REF_HEIGHT) / SLOW_GROWTH_SPAN) * shortfall


def predicted_touchdown_x(s: State) -> float:
    """Horizontal position (m, relative to the pad centre) where the lander would reach the ground if the engines
    cut out now: a ballistic extrapolation of the current velocity under gravity."""
    h = max(s.height, 0.0)
    t = (s.vy + math.sqrt(s.vy * s.vy + 2.0 * C.GRAVITY_MPS2 * h)) / C.GRAVITY_MPS2
    return s.x + s.vx * t


def is_off_course(s: State) -> bool:
    """Airborne and on a path that misses the green zone."""
    return s.legs == 0 and abs(predicted_touchdown_x(s)) > OFFCOURSE_TOLERANCE_M


def crash_position_score(arrival: dict) -> float:
    """Position score of a crash (m miss = distance from the pad centre when it reached the ground).
    Inside the zone: edge value rising exponentially to the centre value. Outside: a large penalty that grows
    exponentially with every metre beyond the edge."""
    zone, d = C.SUCCESS_MAX_DIST_M, arrival["dist_m"]
    if d <= zone:
        bonus = CRASH_IN_EDGE * (CRASH_IN_CENTER / CRASH_IN_EDGE) ** (1.0 - d / zone)
        if bonus > 0 and CRASH_POS_FADE_MPS > 0:       # centring only pays if it is also slow: no credit for a fast crash in the middle
            over = max(0.0, arrival.get("speed_mps", 0.0) - CRASH_SPEED_GOAL)
            bonus *= max(0.0, 1.0 - over / CRASH_POS_FADE_MPS)
        return bonus
    return max(CRASH_OUT_CAP, CRASH_OUT_EDGE * CRASH_OUT_GROWTH ** (d - zone))


def crash_speed_score(speed_mps: float) -> float:
    """Speed score of a crash: 0 at or below the goal speed, then CRASH_SPEED_PER_MPS for every m/s over."""
    return CRASH_SPEED_PER_MPS * max(0.0, speed_mps - CRASH_SPEED_GOAL)


def crash_terms(arrival: dict | None) -> dict:
    """The independent crash scores. arrival = dict(dist_m, tilt_rad, speed_mps) when the lander reached the ground."""
    if arrival is None:
        return dict(crash_pos=0.0, crash_speed=0.0, crash_vert=0.0)
    vert = (CRASH_VERTICAL_BONUS if arrival["dist_m"] <= C.SUCCESS_MAX_DIST_M
            and arrival["tilt_rad"] <= math.radians(VERTICAL_DEG) else 0.0)
    return dict(crash_pos=crash_position_score(arrival), crash_speed=crash_speed_score(arrival.get("speed_mps", 0.0)), crash_vert=vert)


def projected_impact(s: State) -> dict:
    """Where and how fast the lander would hit the ground if the engines cut out now (ballistic path, gravity only)."""
    h = max(s.height, 0.0)
    vy_hit = math.sqrt(s.vy * s.vy + 2.0 * C.GRAVITY_MPS2 * h)
    return dict(dist_m=abs(predicted_touchdown_x(s)), tilt_rad=abs(s.angle), speed_mps=math.hypot(s.vx, vy_hit))


def refire_penalty(ignition_number: int) -> float:
    """Penalty (<= 0) for lighting the main engine for the n-th time. The first ignition is free; the second costs
    REFIRE_BASE and every further one doubles (REFIRE_GROWTH): 25, 50, 100 ... 6400 for the tenth."""
    if ignition_number < 2:
        return 0.0
    return -REFIRE_BASE * REFIRE_GROWTH ** (ignition_number - 2)


def is_ascending(s: State) -> bool:
    return s.legs == 0 and s.vy > ASCEND_SPEED


def compute_reward(
    prev: State,
    cur: State,
    fuel_burned: float,
    end: str | None = None,
    fuel_out_event: bool = False,
    landing: dict | None = None,
    ascend_event: bool = False,
    arrival: dict | None = None,
    ignition: int = 0,
    refire_on: bool = True,
) -> tuple[float, dict]:
    """One step. end: None | 'rest' | 'crash' | 'oob' | 'timeout' | 'tilt' | 'stall'. landing: the touchdown summary, used
    only if end == 'rest'. arrival: where, how tilted and how fast the lander reached the ground (used for a crash, leaving the bounds, or the projected crash of a tilt-out or stall). ignition: the number of the main-engine ignition that started this step (0 if none). ascend_event: the lander started ascending (the caller makes it once per episode).
    Returns (unscaled total, per-term dict)."""
    p0, p1 = potential_parts(prev), potential_parts(cur)
    t = {f"s_{k}": p1[k] - p0[k] for k in p0}                       # shaping, split by cause (for logging)
    t.update(time=TIME_PENALTY, fuel=FUEL_PER_UNIT * fuel_burned,
             hover=-slow_rate_per_sec(cur) / C.FPS,
             ascend=ASCEND_PENALTY if ascend_event else 0.0, tilt_out=0.0, stall_out=0.0,
             refire=refire_penalty(ignition) if refire_on else 0.0,
             first_fire=FIRST_IGNITION_BONUS if ignition == 1 else 0.0,
             offcourse=OFFCOURSE_PER_SEC / C.FPS if is_off_course(cur) else 0.0,
             land=0.0, soft=0.0, crash_pos=0.0, crash_speed=0.0, crash_vert=0.0, crash_pen=0.0, timeout=0.0, fuel_out=0.0)
    if end == "rest" and landing is not None:
        t.update(landing_terms(landing))
    elif end in ("crash", "oob"):
        t.update(crash_terms(arrival))
        t["crash_pen"] = CRASH_PENALTY
    elif end == "timeout":
        t["timeout"] = TIMEOUT_PENALTY
    elif end == "tilt":
        t["tilt_out"] = TILT_OUT_PENALTY
        if TILT_OUT_PROJECTED:
            t.update(crash_terms(arrival))
    elif end == "stall":
        t["stall_out"] = STALL_OUT_PENALTY
        if STALL_OUT_PROJECTED:
            t.update(crash_terms(arrival))
    if fuel_out_event:
        t["fuel_out"] = FUEL_OUT_PENALTY
    return sum(t.values()), t
