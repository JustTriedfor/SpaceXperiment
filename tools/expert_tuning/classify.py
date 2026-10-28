import json, sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np


def one(a):
    params, level, seed = a
    from src.sim import expert as E
    from src.sim.expert import rollout
    from src.sim.spawn import Wind, make_spawn
    from src.utils import config as C
    for k, v in params.items():
        setattr(E, k, v)
    rng = np.random.default_rng(seed)
    w = Wind(C.LEVELS[level]["wind_power"], rng)
    s, d = make_spawn(rng, level, prop=C.FUEL_BUDGET, wind=w)
    r = rollout(s, w)
    if r["t"] >= 119: return level, "never landed", r
    if r["out_of_prop"]: return level, "out of propellant", r
    if r["tilt"] > 0.2 or r["speed"] > 5 or abs(r["vx"]) > 3: return level, "CRASH", r
    if abs(r["x"]) > 10: return level, "missed zone (>10 m)", r
    if r["speed"] > 3 or r["tilt"] > 0.15: return level, "not soft", r
    return level, "SOFT", r


if __name__ == "__main__":
    params = json.load(open(sys.argv[1])) if sys.argv[1] != "default" else {}
    with Pool(12) as p:
        out = p.map(one, [(params, lv, 900_000 + 1000 * lv + i) for lv in (0, 1, 2) for i in range(200)], chunksize=4)
    for lv in (0, 1, 2):
        rows = [(c, r) for l, c, r in out if l == lv]
        print(lv, dict(Counter(c for c, _ in rows)), "median miss", np.median([abs(r["x"]) for c, r in rows if c != "never landed"]).round(1),
              "median speed", np.median([r["speed"] for c, r in rows if c != "never landed"]).round(2), "median |vx|", np.median([abs(r["vx"]) for c, r in rows if c != "never landed"]).round(2),
              "median prop used", np.median([r["prop_used"] for c, r in rows]).round(0))
