"""The vehicle: a Falcon 9 first stage on its landing burn. Every number here is an APPROXIMATE public figure (or a documented
engineering estimate, marked "estimate") - SpaceX does not publish its flight data. SI units; angles in rad.

Sources, in the order they are used:
  Block 5 booster dry mass ~22.2 t, length ~41.2 m, diameter 3.66 m (SpaceX user's guide / public references).
  Merlin 1D: ~845 kN at sea level, ~914 kN in vacuum, Isp ~282 s sea level / ~311 s vacuum, throttles down to ~40%.
  The landing burn is flown on one engine; touchdown speeds of 1-2 m/s have been reported for good landings.
Anything marked "estimate" is the author's modelling choice and is meant to be recalibrated (see src/sim/calibrate.py).
"""
from __future__ import annotations

import math

from src.utils import config as C

G0 = 9.80665                      # standard gravity, for Isp -> mass flow

# ---- mass and geometry
DRY_MASS = 22_200.0               # kg
LENGTH = 41.2                     # m (first stage)
DIAMETER = 3.66                   # m
CG_BASE = 14.0                    # m of the centre of gravity above the engine bells (estimate: engines are heavy, tanks above)
FOOT_BELOW_BASE = 1.0             # m: the deployed leg feet are this far below the engine bells
LEG_STIFFNESS = 5.0e5             # N/m per foot: the leg's shock absorber (estimate); the stage settles about 0.25 m onto its legs
LEG_DAMPING = 1.2e5               # N s/m per foot: about 0.75 of critical for half the stage on one leg (estimate)
LEG_STROKE = 0.9                  # m of leg travel before it bottoms out (estimate)
LEG_BOTTOM_STIFFNESS = 8.0e6      # N/m once it has bottomed out
FOOT_FRICTION = 0.6               # sliding friction of a foot on the deck
LEG_SPAN = 18.0                   # m between the two feet shown in the 2-D plane (deployed legs, about 18 m)
BASE_AREA = math.pi * (DIAMETER / 2) ** 2          # m^2 frontal area when flying engine-first
SIDE_AREA = DIAMETER * LENGTH                      # m^2 projected area in cross-flow
INERTIA_PER_KG = LENGTH ** 2 / 12.0 + (LENGTH / 2.0 - CG_BASE) ** 2    # m^2: I = m x this (uniform rod about the CG; estimate)

# ---- main engine (one Merlin 1D)
THRUST_VAC = 914_000.0            # N
THRUST_SL = 845_000.0             # N
ISP_VAC = 311.0                   # s
THROTTLE_MIN = 0.40               # fraction of full thrust when running (cannot go lower; cannot hover)
MDOT_FULL = THRUST_VAC / (ISP_VAC * G0)            # kg/s at full throttle (~300)
PRESSURE_SCALE_HEIGHT = 7200.0    # m (thrust rises from sea-level to vacuum value as ambient pressure drops)
SPOOL_TAU = 0.25                  # s, first-order lag of the thrust when the command rises (estimate)
SHUTDOWN_TAU = 0.10               # s, when it falls
GIMBAL_MAX = math.radians(5.0)    # thrust-vector control range of the centre engine
GIMBAL_TAU = 0.10                 # s (estimate)
GIMBAL_RATE = math.radians(20.0)  # rad/s

# ---- grid fins (modelled as one lifting pair in the 2-D plane)
FIN_MAX = math.radians(25.0)
FIN_TAU = 0.08
FIN_RATE = math.radians(60.0)
FIN_AREA = 6.0                    # m^2 effective total area (estimate)
FIN_CL_ALPHA = 2.0                # per rad of deflection (estimate; grid fins are very effective at high dynamic pressure)
FIN_ARM = 24.0                    # m above the CG (the fins sit near the top of the stage)

# ---- cold-gas RCS: two independent systems (estimate: real boosters carry nitrogen thrusters near the top)
RCS_FORCE = 5_000.0               # N per system at full command
RCS_ARM_A = 21.0                  # m above the CG (system A: upper, strongest torque)
RCS_ARM_B = -11.0                 # m below the CG (system B: near the engine skirt; acts against A for translation)
RCS_ISP = 70.0                    # s (cold nitrogen)
RCS_GAS = 90.0                    # kg of nitrogen
RCS_DEADBAND = 0.10

# ---- atmosphere and aerodynamics (estimates)
RHO0 = 1.225                      # kg/m^3 at sea level
DENSITY_SCALE_HEIGHT = 8400.0     # m
CD_AXIAL_BASE_FIRST = 1.2         # engine-first (the landing attitude): a blunt base with engines and legs
CD_AXIAL_NOSE_FIRST = 0.55
CD_CROSS = 1.1                    # flow across the body (cylinder)
CROSS_STATIONS = tuple(z - CG_BASE for z in (LENGTH / 6.0, LENGTH / 2.0, 5.0 * LENGTH / 6.0))   # cross-flow slices: heights above the CG. Their mean is above the CG, so engine-first flight is weathercock-stable.

# apply web-app / command-line overrides (name -> value) for any constant defined above
for _k, _v in C.OVERRIDES.get("vehicle", {}).items():
    if _k in globals() and _k.isupper():
        globals()[_k] = _v


def density(h_m: float) -> float:
    return RHO0 * math.exp(-max(h_m, 0.0) / DENSITY_SCALE_HEIGHT)


def thrust_full(h_m: float) -> float:
    """Thrust (N) at full throttle at this altitude: vacuum value minus the ambient-pressure loss."""
    return THRUST_VAC - (THRUST_VAC - THRUST_SL) * math.exp(-max(h_m, 0.0) / PRESSURE_SCALE_HEIGHT)


def inertia(mass: float) -> float:
    return mass * INERTIA_PER_KG


def terminal_velocity(mass: float, h_m: float = 0.0) -> float:
    """Engine-first terminal speed (m/s) at this altitude: the speed at which drag equals weight."""
    return math.sqrt(2.0 * mass * C.GRAVITY_MPS2 / (density(h_m) * CD_AXIAL_BASE_FIRST * BASE_AREA))
