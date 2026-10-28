"""How well does the scripted expert land?   python -m src.sim.eval_expert [--n 60] [--levels 0,1,2,3] [--workers 8]

Per level: success (within the zone, soft, upright), survival, and WHY it failed (too fast, too far, tilted, stopped above the ground, ...).
"""
from __future__ import annotations

import argparse
from collections import Counter
from multiprocessing import Pool

import numpy as np

from src.sim.expert import rollout
from src.sim.spawn import Wind, make_spawn
from src.utils import config as C


def one(args):
    level, seed = args
    rng = np.random.default_rng(seed)
    w = Wind(C.LEVELS[level]["wind_power"], rng)
    s, d = make_spawn(rng, level, prop=C.FUEL_BUDGET, wind=w)
    r = rollout(s, w)
    return level, r


def classify(r) -> str:
    if r["t"] >= 119:
        return "never landed (climbed or hovered)"
    if r["tilt"] > C.SURVIVE_MAX_TILT_RAD:
        return "crash: tilted"
    if r["speed"] > C.SURVIVE_MAX_VSPEED_MPS:
        return "crash: too fast"
    if abs(r["vx"]) > C.SURVIVE_MAX_HSPEED_MPS:
        return "crash: sideways speed"
    if r["out_of_prop"]:
        return "survived but out of propellant"
    if abs(r["x"]) > C.SUCCESS_MAX_DIST_M:
        return "survived, missed the zone"
    if r["speed"] > C.SUCCESS_MAX_SPEED_MPS or r["tilt"] > C.SUCCESS_MAX_TILT_RAD:
        return "survived, not soft"
    return "SUCCESS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--levels", default="0,1,2,3")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()
    levels = [int(x) for x in a.levels.split(",")]
    jobs = [(lv, a.seed + 1000 * lv + i) for lv in levels for i in range(a.n)]
    with Pool(a.workers) as pool:
        out = pool.map(one, jobs, chunksize=2)
    for lv in levels:
        rs = [r for l, r in out if l == lv]
        cats = Counter(classify(r) for r in rs)
        ok = [r for r in rs if classify(r) == "SUCCESS"]
        used = [r["prop_used"] for r in rs if r["t"] < 119]
        spd = [r["speed"] for r in rs if r["t"] < 119]
        miss = [abs(r["x"]) for r in rs if r["t"] < 119]
        print(f"level {lv}: success {len(ok)}/{len(rs)} ({100 * len(ok) / len(rs):.0f}%) | median speed {np.median(spd):.1f} m/s, miss {np.median(miss):.0f} m, "
              f"propellant {np.median(used):.0f} kg")
        for k, v in cats.most_common():
            print(f"     {v:3d}  {k}")


if __name__ == "__main__":
    main()
