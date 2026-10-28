"""Start states: the re-entry side entry.

The booster arrives at SPAWN_HEIGHT_M (3 km) at up to ~300 m/s on a descending path from one side, engine-first with a small angle of attack.
The start position is found by SIMULATING a nominal landing (coast with drag, then a straight braking burn with no sideways steering) from the
origin and shifting the start so that this path would end near the pad (within the level's miss allowance). The agent then has to steer in, brake and land.
"""
from __future__ import annotations

import math

import numpy as np

from src.sim import vehicle as V
from src.sim.physics import State, passive_descent, powered_descent
from src.utils import config as C


class Wind:
    """Wind speed along x: a per-episode mean plus a slow gust, stronger aloft (shear). Positive = blowing toward +x."""

    def __init__(self, power: float = 0.0, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        self.mean = float(rng.uniform(-1.0, 1.0)) * 0.6 * power
        self.amp = 0.4 * power
        self.period = float(rng.uniform(6.0, 14.0))
        self.phase = float(rng.uniform(0.0, 2.0 * math.pi))

    def at(self, t: float, h: float) -> float:
        shear = 0.4 + 0.6 * min(max(h, 0.0) / 1500.0, 1.0)
        return shear * (self.mean + self.amp * math.sin(2.0 * math.pi * t / self.period + self.phase))


def entry_range(cfg) -> tuple:
    """The (min, max) entry angle in degrees: the chosen preset's, or the level's own when the preset is "normal"."""
    key = str(C.ENTRY_PRESET).lower().replace(" ", "").replace("_", "").replace("-", "")
    key = {"straightdown": "straight"}.get(key, key)
    return C.ENTRY_PRESETS.get(key, cfg["entry_deg"])


def make_spawn(rng: np.random.Generator, level: int, prop: float | None = None, height_m: float | None = None, wind: Wind | None = None):
    """-> (State, details). details: speed, entry_deg, side, target_miss_m, predicted_impact_x."""
    cfg = C.LEVELS[level]
    prop = C.FUEL_BUDGET if prop is None else prop
    height = C.SPAWN_HEIGHT_M if height_m is None else height_m
    var = min(1.0, max(0.0, float(C.TRAJ_VARIATION)))
    pick = lambda a, b: a + (b - a) * (0.5 + (float(rng.random()) - 0.5) * var)      # uniform in [a, b] at var = 1, always the middle at var = 0
    speed = pick(C.SPACEX_V_MIN, C.SPACEX_V_MAX) * cfg["speed"] * C.SPACEX_LEVEL_SCALE[level] * C.SPACEX_SPEED_SCALE
    lo, hi = entry_range(cfg)
    gamma = math.radians(pick(lo, hi))
    side = float(rng.choice([-1.0, 1.0]))                      # +1: arriving from the left, moving toward +x
    vx, vy = side * speed * math.cos(gamma), -speed * math.sin(gamma)
    aoa = float(rng.uniform(-1.0, 1.0)) * math.radians(cfg["aoa_deg"])
    cap = math.radians(C.TILT_SPAWN_MAX_DEG)
    th = max(-cap, min(cap, math.atan2(vx, -vy) + aoa))         # engine toward the oncoming air, plus a small angle of attack
    om = float(rng.uniform(-1.0, 1.0)) * cfg["ang_vel"]
    s = State(x=0.0, y=height + C.LEG_DOWN_M, vx=vx, vy=vy, th=th, om=om, prop=prop)
    s.y += height - s.foot_height()                                # the lower foot starts exactly `height` above the pad, whatever the tilt
    w0 = wind.at(0.0, height) if wind else 0.0
    impact_v = passive_descent(s, wind_x=w0)[1]
    impact_x = powered_descent(s, wind_x=w0)[0]                 # where a straight, normal landing burn would end up (not the passive path: the burn lasts ~9 s more)
    miss = float(np.clip(rng.normal(0.0, cfg["x_offset"] / 2.0), -cfg["x_offset"], cfg["x_offset"]))
    s.x = miss - impact_x                                       # shift the whole path so that it would end `miss` metres from the pad centre
    return s, dict(speed=speed, entry_deg=math.degrees(gamma), side=side, target_miss_m=miss, passive_impact_speed=impact_v, wind0=w0)
