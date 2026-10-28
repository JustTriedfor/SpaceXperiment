"""Hill-climb the scripted expert's constants to cut its crash rate at levels 1 and 2 (and keep level 0 perfect).
   python tune_expert.py <iterations> <workers> <out.json>
A crash = touchdown with vertical speed > 5 m/s, sideways speed > 3 m/s or tilt > 0.2 rad, or never landing (the survive limits of the environment).
"""
import json, math, sys, time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

SPACE = dict(                       # name: (low, high, log-scale?)
    A_PLAN_FRACTION=(0.5, 0.95, False), LATERAL_TILT_MAX=(0.2, 0.7, False), LATERAL_TILT_GROUND=(0.02, 0.2, False),
    LATERAL_TILT_PER_M=(0.001, 0.012, True), LATERAL_VX_GAIN=(1.0, 12.0, True), LATERAL_TGO_SCALE=(0.3, 1.4, False),
    LATERAL_TGO_MIN=(1.0, 6.0, False), LATERAL_X_FADE_M=(40.0, 500.0, True), LATERAL_LEAD_S=(0.0, 3.0, False),
    ALIGN_RAMP_M=(300.0, 2500.0, True), ATT_WN_MAX=(0.5, 3.0, False), V_TOUCH=(1.0, 2.2, False),
)
LEVELS = (0, 1, 2)
N = {0: 60, 1: 300, 2: 200}
W = {0: 2.0, 1: 1.0, 2: 0.4}


def one(args):
    params, level, seed = args
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
    landed = r["t"] < 119
    crash = (not landed) or r["speed"] > 5.0 or abs(r["vx"]) > 3.0 or r["tilt"] > 0.2
    soft = landed and not crash and abs(r["x"]) <= 10 and r["speed"] <= 3.0 and r["tilt"] <= 0.15 and not r["out_of_prop"]
    return level, crash, soft


def evaluate(pool, params, seed0):
    jobs = [(params, lv, seed0 + 1000 * lv + i) for lv in LEVELS for i in range(N[lv])]
    res = pool.map(one, jobs, chunksize=4)
    out = {}
    for lv in LEVELS:
        rows = [r for r in res if r[0] == lv]
        out[lv] = (sum(r[1] for r in rows) / len(rows), sum(r[2] for r in rows) / len(rows))
    obj = sum(W[lv] * ((1.0 - out[lv][1]) + 2.0 * out[lv][0]) for lv in LEVELS)       # miss the soft landing in the zone: cost 1; crash: cost 3
    return obj, out


def defaults():
    from src.sim import expert as E
    return {k: getattr(E, k) for k in SPACE}


def clip(k, v):
    lo, hi, _ = SPACE[k]
    return min(hi, max(lo, v))


def main():
    iters, workers, outp = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rng = np.random.default_rng(1)
    best = defaults()
    if len(sys.argv) > 4:
        best.update(json.load(open(sys.argv[4])))
    with Pool(workers) as pool:
        t0 = time.time()
        bobj, bout = evaluate(pool, best, 10_000)
        print(f"start  obj {bobj:.4f}  crash L0/L1/L2 = {bout[0][0]:.3f}/{bout[1][0]:.3f}/{bout[2][0]:.3f}  soft {bout[0][1]:.2f}/{bout[1][1]:.2f}/{bout[2][1]:.2f}", flush=True)
        for it in range(iters):
            cand = dict(best)
            for k in rng.choice(list(SPACE), size=int(rng.integers(1, 4)), replace=False):
                lo, hi, lg = SPACE[k]
                if lg:
                    cand[k] = clip(k, cand[k] * math.exp(rng.normal(0, 0.35)))
                else:
                    cand[k] = clip(k, cand[k] + rng.normal(0, 0.18 * (hi - lo)))
            obj, out = evaluate(pool, cand, 10_000)
            if obj < bobj:
                best, bobj, bout = cand, obj, out
                print(f"it {it:3d} NEW obj {bobj:.4f}  crash {bout[0][0]:.3f}/{bout[1][0]:.3f}/{bout[2][0]:.3f}  soft {bout[0][1]:.2f}/{bout[1][1]:.2f}/{bout[2][1]:.2f}  ({time.time()-t0:.0f}s)", flush=True)
                json.dump(best, open(outp, "w"), indent=1)
        # validation on fresh seeds
        vobj, vout = evaluate(pool, best, 500_000)  # fresh seeds
        print(f"VALIDATION (fresh seeds)  crash {vout[0][0]:.3f}/{vout[1][0]:.3f}/{vout[2][0]:.3f}  soft {vout[0][1]:.2f}/{vout[1][1]:.2f}/{vout[2][1]:.2f}", flush=True)
        json.dump(best, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
