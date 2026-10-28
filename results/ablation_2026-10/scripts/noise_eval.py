"""Job 3: sensor noise.  The trained controller (rt1_final, NOT retrained) flies seeds 0-299 at L0-L3 with Gaussian noise added to the observation vector the POLICY sees
(the simulator, its dynamics and the success rule are untouched; the noise is added outside the environment, at the policy input).

    python results/ablation_2026-10/scripts/noise_eval.py [--workers 12]

Noise level q in {0, 1, 2, 5} % means: each observation i gets N(0, (q/100 * span_i)^2) added at every control step (20 Hz), independent per step and per observation, not clipped.
span_i = (max - min) of observation i over the CLEAN flights of the same controller on the VALIDATION seeds 100000-100199, all four levels (data/obs_ranges.json). It is not the declared
Box(-20, 20) of the environment, which would make 1 % a noise of 0.4 (the observations are scaled to roughly [-1, 1]).
The noise draws of an episode come from default_rng([seed, level, 31337]): the same standard-normal sequence is scaled by every q (common random numbers); q = 0 is the clean run.
"""
import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CHECKPOINT, OUT, ROOT, wilson  # noqa: E402

sys.path.insert(0, str(ROOT))
LEVELS = (0, 1, 2, 3)
PCTS = (0, 1, 2, 5)
_M = {}


def _init():
    import torch
    torch.set_num_threads(1)


def model():
    if "m" not in _M:
        from stable_baselines3 import PPO
        from src.training.weights_io import resolve_checkpoint
        _M["m"] = PPO.load(str(resolve_checkpoint(CHECKPOINT)), device="cpu")
    return _M["m"]


def fly(level, seed, pct, span):
    from src.sim.env import SpaceXEnv
    env = SpaceXEnv(level=level)
    env.set_rules_stage(2)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng([seed, level, 31337])
    done, seen = False, []
    while not done:
        z = rng.normal(0.0, 1.0, obs.shape).astype(np.float32)
        seen.append(obs.copy())
        act = model().predict(obs + (pct / 100.0) * span * z if pct else obs, deterministic=True)[0]
        obs, _, te, tr, info = env.step(act)
        done = te or tr
    m = info["episode_metrics"]
    return m, np.array(seen)


def calib_job(args):
    level, seed = args
    _, seen = fly(level, seed, 0, None)
    return seen.min(axis=0), seen.max(axis=0)


def eval_job(args):
    pct, level, seed, span = args
    m, _ = fly(level, seed, pct, np.asarray(span, dtype=np.float32))
    return pct, level, seed, int(bool(m["success"])), float(m["speed_mps"]) if m["touchdown"] else float("nan"), float(m["dist_m"]) if m["touchdown"] else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        cal = pool.map(calib_job, [(lv, 100_000 + i) for lv in LEVELS for i in range(200)], chunksize=4)
        lo = np.min([c[0] for c in cal], axis=0)
        hi = np.max([c[1] for c in cal], axis=0)
        span = hi - lo
        json.dump(dict(min=lo.tolist(), max=hi.tolist(), span=span.tolist(), seeds="100000-100199, levels 0-3, clean flights of " + CHECKPOINT), open(OUT / "data" / "obs_ranges.json", "w"), indent=1)
        print("observation spans:", np.round(span, 3).tolist(), f"({time.time() - t0:.0f} s)", flush=True)
        jobs = [(p, lv, s, span.tolist()) for p in PCTS for lv in LEVELS for s in range(300)]
        rows = pool.map(eval_job, jobs, chunksize=4)
    with open(OUT / "data" / "job3_noise_episodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["noise_percent", "level", "seed", "success", "touchdown_speed_mps", "pad_offset_m"])
        w.writerows(sorted(rows))
    res, md = {}, ["# Job 3: sensor noise on the observations (trained controller, not retrained, seeds 0-299, 300 episodes per cell)", "",
                   "Gaussian noise added to every observation at every step, standard deviation = noise % of that observation's range (range measured on clean flights, validation seeds; see `data/obs_ranges.json`). "
                   "Cells: successes / 300, success rate [95 % Wilson interval].", "", "| noise | L0 | L1 | L2 | L3 |", "|---|---|---|---|---|"]
    for p in PCTS:
        cells = []
        for lv in LEVELS:
            k = sum(r[3] for r in rows if r[0] == p and r[1] == lv)
            l, h = wilson(k, 300)
            res[f"noise{p}_L{lv}"] = dict(noise_percent=p, level=lv, n=300, successes=k, rate=k / 300, wilson_lower=l, wilson_upper=h)
            cells.append(f"{k}/300 = {k / 300:.3f} [{l:.3f}, {h:.3f}]")
        md.append(f"| {p} % | " + " | ".join(cells) + " |")
    clean = {lv: res[f"noise0_L{lv}"]["successes"] for lv in LEVELS}
    md += ["", f"Noise 0 % is the clean run: {clean[0]}, {clean[1]}, {clean[2]}, {clean[3]} successes (the paper: 300, 259, 161, 90).", "",
           "Change in success rate against 0 % (percentage points):", "", "| noise | L0 | L1 | L2 | L3 |", "|---|---|---|---|---|"]
    for p in PCTS[1:]:
        md.append(f"| {p} % | " + " | ".join(f"{100 * (res[f'noise{p}_L{lv}']['rate'] - res[f'noise0_L{lv}']['rate']):+.1f}" for lv in LEVELS) + " |")
    (OUT / "job3_sensor_noise.md").write_text("\n".join(md) + "\n")
    json.dump(res, open(OUT / "data" / "job3.json", "w"), indent=1)
    print("\n".join(md), f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
