"""Merit score: how good was a landing, 0-100, computed the same way for every agent.

Independent of the training reward (use it to compare agents, never their reward values).
  0                       unless the lander touched down AND the episode ended at rest (a survived landing)
  otherwise 100 x ( 0.35 softness + 0.20 centering + 0.15 uprightness + 0.10 fuel left + 0.20 single burn )
    softness   = max(0, 1 - touchdown_speed / 2 m/s)      (full marks only near 0 m/s; the success limit is 1.5 m/s)
    centering  = max(0, 1 - miss / 2 m)                   (the pad radius)
    uprightness= max(0, 1 - tilt / 0.35 rad)
    fuel left  = fraction of the fuel budget unused
    single burn= 1 for one engine ignition, halved for every extra ignition (a real Falcon 9 lights its engine once)
Also here: the "soft landing" used to gate the curriculum. Both it and the strict success now use a 1.5 m/s touchdown limit.
"""
from __future__ import annotations

from src.utils import config as C


def merit(touchdown: dict | None, end: str, fuel_left: float, ignitions: int = 1) -> float:
    if touchdown is None or end != "rest":
        return 0.0
    soft = max(0.0, 1.0 - touchdown["speed_mps"] / (C.SUCCESS_MAX_SPEED_MPS * 4.0 / 3.0))
    center = max(0.0, 1.0 - touchdown["dist_m"] / C.SUCCESS_MAX_DIST_M)
    upright = max(0.0, 1.0 - touchdown["tilt_rad"] / 0.35)
    one_burn = 1.0 if ignitions <= 1 else 0.5 ** (ignitions - 1)         # a real booster lights its engine once
    return 100.0 * (0.35 * soft + 0.20 * center + 0.15 * upright + 0.10 * max(0.0, min(1.0, fuel_left)) + 0.20 * one_burn)


def is_soft_landing(touchdown: dict | None, end: str) -> bool:
    """Survived, on the pad, upright and at most GATE_MAX_SPEED_MPS at touchdown (curriculum gate)."""
    return (touchdown is not None and end == "rest"
            and touchdown["dist_m"] <= C.SUCCESS_MAX_DIST_M
            and touchdown["tilt_rad"] <= C.SUCCESS_MAX_TILT_RAD
            and touchdown["speed_mps"] <= C.GATE_MAX_SPEED_MPS)
