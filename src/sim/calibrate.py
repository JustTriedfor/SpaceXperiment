"""Calibration of the vehicle and of the fuel budget.   python -m src.sim.calibrate [--n 60]

1. Engine / mass table: thrust-to-weight at full and minimum throttle, net braking capacity, hover check, terminal velocities.
2. Fuel: the scripted expert (src/sim/expert.py) flies the spawn distribution of every level with plenty of propellant; the propellant of the
   landings that survive gives the need. Budget = FUEL_MARGIN x the median need of the survivors (the rule used in the old project: 1.5x).
Writes models/configs/calibration_spacexperiment.json.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from src.sim import vehicle as V
from src.sim.expert import rollout
from src.sim.spawn import Wind, make_spawn
from src.utils import config as C

FUEL_MARGIN = 1.5


def engine_table(prop_kg: float) -> dict:
    m = V.DRY_MASS + prop_kg + V.RCS_GAS
    out = {}
    for h, label in ((0.0, "sea level"), (3000.0, "3 km")):
        t = V.thrust_full(h)
        out[label] = dict(thrust_full_kN=round(t / 1e3), twr_full=round(t / (m * C.GRAVITY_MPS2), 2), twr_min=round(V.THROTTLE_MIN * t / (m * C.GRAVITY_MPS2), 2),
                          net_brake_full_mps2=round(t / m - C.GRAVITY_MPS2, 1), net_brake_min_mps2=round(V.THROTTLE_MIN * t / m - C.GRAVITY_MPS2, 1))
    out["can_hover"] = V.THROTTLE_MIN * V.thrust_full(0.0) < m * C.GRAVITY_MPS2
    out["mass_kg"] = round(m)
    out["terminal_velocity_mps"] = {f"{h} m": round(V.terminal_velocity(m, h)) for h in (0, 1500, 3000)}
    out["mdot_full_kg_s"] = round(V.MDOT_FULL, 1)
    out["burn_time_full_s_per_tonne"] = round(1000.0 / V.MDOT_FULL, 2)
    return out


def fuel_need(n: int, levels=(0, 1, 2, 3), seed: int = 0, prop: float = 9000.0) -> dict:
    rng = np.random.default_rng(seed)
    res = {}
    for level in levels:
        used, ok, surv = [], 0, 0
        for _ in range(n):
            w = Wind(C.LEVELS[level]["wind_power"], rng)
            s, _ = make_spawn(rng, level, prop=prop, wind=w)
            r = rollout(s, w)
            alive = (r["speed"] <= C.SURVIVE_MAX_VSPEED_MPS and r["tilt"] <= C.SURVIVE_MAX_TILT_RAD and abs(r["vx"]) <= C.SURVIVE_MAX_HSPEED_MPS and r["t"] < 119)
            good = alive and r["speed"] <= C.SUCCESS_MAX_SPEED_MPS and r["tilt"] <= C.SUCCESS_MAX_TILT_RAD and abs(r["x"]) <= C.SUCCESS_MAX_DIST_M
            surv += alive
            ok += good
            if alive:
                used.append(r["prop_used"])
        res[level] = dict(n=n, survived=surv, succeeded=ok, median_kg=round(float(np.median(used))) if used else None,
                          p90_kg=round(float(np.percentile(used, 90))) if used else None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    table = engine_table(C.FUEL_BUDGET)
    need = fuel_need(a.n, seed=a.seed)
    meds = [v["median_kg"] for v in need.values() if v["median_kg"]]
    budget = round(FUEL_MARGIN * float(np.median(meds)), -1) if meds else C.FUEL_BUDGET
    report = dict(vehicle=table, expert=need, margin=FUEL_MARGIN, recommended_fuel_budget_kg=budget, current_fuel_budget_kg=C.FUEL_BUDGET,
                  note="Approximate public figures; see src/sim/vehicle.py for sources and which numbers are estimates.")
    C.MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (C.MODEL_CONFIG_DIR / "calibration_spacexperiment.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
