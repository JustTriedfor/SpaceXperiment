"""Was the expert's tuning held back by its search limits?  (new script; src/ and tools/ are not changed, nothing is written outside this folder)

    python paper_assets/expert_failures/tune_wider.py <iterations> <workers> [original]

Two of the 12 tuned constants ended on the edge of the range tools/expert_tuning/tune_expert.py allowed: LATERAL_TILT_MAX (limit 0.7) and ALIGN_RAMP_M (limit 2500).
This repeats the same random hill-climb (same objective, same tuner seeds 10,000+, same step sizes, start = the expert as it is now) with those two limits widened
(LATERAL_TILT_MAX up to 1.2 rad, ALIGN_RAMP_M up to 6000 m).  The other ten limits are the original ones.
Then, once, on fresh seeds 500,000+ (the tuner's own validation) and on the paper's test seeds 0-299 (expert_tuned rule set), the start and the best result are compared.
The test seeds are used ONLY for this final comparison, never to choose anything.  Writes tune_wider.log-style output to stdout, tune_wider_best.json and tune_wider.md.
"""
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "expert_tuning"))
import tune_expert as T  # noqa: E402

ORIGINAL = len(sys.argv) > 3 and sys.argv[3] == "original"      # control run: the same search with the ORIGINAL limits, to separate "wider limits" from "more iterations"
TAG = "tune_original_limits" if ORIGINAL else "tune_wider"
if not ORIGINAL:
    T.SPACE["LATERAL_TILT_MAX"] = (0.2, 1.2, False)
    T.SPACE["ALIGN_RAMP_M"] = (300.0, 6000.0, True)


def test_flight(args):
    """one test-seed episode with the paper's loop (same as paper_assets/scripts/run_eval.py, expert_tuned, noise 0)"""
    params, level, seed = args
    from src.sim import expert as E
    for k, v in params.items():
        setattr(E, k, v)
    from src.sim.env import SpaceXEnv
    from src.sim.expert_data import _label
    env = SpaceXEnv(level=level)
    env.set_rules_stage(2)
    env.reset(seed=seed)
    ex = E.Expert()
    ex.reset()
    done = False
    while not done:
        _, _, te, tr, info = env.step(_label(ex, env, False))
        done = te or tr
    m = info["episode_metrics"]
    return level, bool(m["success"]), float(m["dist_m"]), float(m["speed_mps"])


def test_eval(pool, params):
    res = pool.map(test_flight, [(params, lv, sd) for lv in range(4) for sd in range(300)], chunksize=6)
    return {lv: (sum(r[1] for r in res if r[0] == lv), float(np.median([r[2] for r in res if r[0] == lv and not r[1]] or [0.0]))) for lv in range(4)}


def main():
    iters, workers = int(sys.argv[1]), int(sys.argv[2])
    rng = np.random.default_rng(1)                                # same generator seed as the original tuner
    start = T.defaults()
    best = dict(start)
    t0 = time.time()
    with Pool(workers) as pool:
        bobj, bout = T.evaluate(pool, best, 10_000)
        sobj = bobj
        print(f"start  obj {bobj:.4f}  crash L0/L1/L2 = {bout[0][0]:.3f}/{bout[1][0]:.3f}/{bout[2][0]:.3f}  soft {bout[0][1]:.2f}/{bout[1][1]:.2f}/{bout[2][1]:.2f}", flush=True)
        for it in range(iters):
            cand = dict(best)
            for k in rng.choice(list(T.SPACE), size=int(rng.integers(1, 4)), replace=False):
                lo, hi, lg = T.SPACE[k]
                if lg:
                    cand[k] = T.clip(k, cand[k] * math.exp(rng.normal(0, 0.35)))
                else:
                    cand[k] = T.clip(k, cand[k] + rng.normal(0, 0.18 * (hi - lo)))
            obj, out = T.evaluate(pool, cand, 10_000)
            if obj < bobj:
                best, bobj, bout = cand, obj, out
                print(f"it {it:3d} NEW obj {bobj:.4f}  crash {bout[0][0]:.3f}/{bout[1][0]:.3f}/{bout[2][0]:.3f}  soft {bout[0][1]:.2f}/{bout[1][1]:.2f}/{bout[2][1]:.2f}  "
                      f"TILT_MAX {best['LATERAL_TILT_MAX']:.3f} ALIGN_RAMP {best['ALIGN_RAMP_M']:.0f}  ({time.time() - t0:.0f}s)", flush=True)
                json.dump(best, open(OUT / f"{TAG}_best.json", "w"), indent=1)
            elif it % 25 == 24:
                print(f"it {it:3d} (no better yet, {time.time() - t0:.0f}s)", flush=True)
        json.dump(best, open(OUT / f"{TAG}_best.json", "w"), indent=1)
        vs, vb = T.evaluate(pool, start, 500_000), T.evaluate(pool, best, 500_000)
        ts, tb = test_eval(pool, start), test_eval(pool, best)
    f = lambda o: f"{o[1][0]:.3f}/{o[1][1]:.3f}/{o[1][2]:.3f}"
    md = [("# Control: tuning again with the ORIGINAL limits" if ORIGINAL else "# Tuning again with the two edge limits widened") + " (new run; the expert in src/ is unchanged)", "",
          f"Iterations: {iters}; same objective, seeds and steps as `tools/expert_tuning/tune_expert.py`; start = the expert as it is now. " + ("Limits as in the original tuner." if ORIGINAL else "Widened: LATERAL_TILT_MAX up to 1.2 rad (was 0.7), ALIGN_RAMP_M up to 6000 m (was 2500)."), "",
          "| constant | start (current expert) | best found |", "|---|---|---|"]
    for k in T.SPACE:
        md.append(f"| {k} | {start[k]:.5g} | {best[k]:.5g} |")
    md += ["", "## Tuner objective (lower is better) and results on the tuner's own seeds 10,000+", "", f"start {sobj:.4f}, best {bobj:.4f}.", "",
           "## Fresh seeds 500,000+ (the tuner's validation set): crash rate L0/L1/L2 and soft-landing rate", "",
           f"| | crash L0/L1/L2 | soft L0/L1/L2 |", "|---|---|---|",
           f"| start | {vs[1][0][0]:.3f}/{vs[1][1][0]:.3f}/{vs[1][2][0]:.3f} | {vs[1][0][1]:.3f}/{vs[1][1][1]:.3f}/{vs[1][2][1]:.3f} |",
           f"| best | {vb[1][0][0]:.3f}/{vb[1][1][0]:.3f}/{vb[1][2][0]:.3f} | {vb[1][0][1]:.3f}/{vb[1][1][1]:.3f}/{vb[1][2][1]:.3f} |", "",
           "## Paper test seeds 0-299 (looked at once, after the search; paper success rule), successes of 300 and median miss of the failures", "",
           "| | L0 | L1 | L2 | L3 |", "|---|---|---|---|---|",
           "| start | " + " | ".join(f"{ts[lv][0]} (miss {ts[lv][1]:.1f} m)" for lv in range(4)) + " |",
           "| best | " + " | ".join(f"{tb[lv][0]} (miss {tb[lv][1]:.1f} m)" for lv in range(4)) + " |"]
    (OUT / f"{TAG}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
