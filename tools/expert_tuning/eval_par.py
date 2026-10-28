"""Parallel evaluation of a pair checkpoint (or the scripted expert) on Dual land.
   python eval_par.py <pair-name|EXPERT> <episodes-per-level> <levels e.g. 0,1,2> [workers]
Prints, per level: lander-level end counts, pair crash rate (either lander not at rest), per-lander crash rate, joint success, gap.
"""
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np


def run_one(args):
    name, level, seed = args
    import torch
    torch.set_num_threads(1)                      # 16 workers x many torch threads each stalls the whole machine
    from src.sim.dual_env import DualEnv, agent_view
    from src.utils import config as C
    env = DualEnv(level=level)
    env.set_rules_stage(2)
    obs, _ = env.reset(seed=seed)
    done = False
    if name == "EXPERT":
        from src.sim.expert import Expert
        from src.sim.expert_data import _label
        exs = [Expert(), Expert()]
        while not done:
            clean = np.concatenate([_label(exs[k], (env.a, env.b)[k], env.done[k]) for k in (0, 1)])
            obs, _, te, tr, info = env.step(clean)
            done = te or tr
    else:
        from src.training.pair import load_pair_models
        mA, mB = load_pair_models(name)
        while not done:
            act = np.concatenate([mA.predict(agent_view(obs, 0), deterministic=True)[0], mB.predict(agent_view(obs, 1), deterministic=True)[0]])
            obs, _, te, tr, info = env.step(act)
            done = te or tr
    m = info["episode_metrics"]
    return level, list(env.end), bool(m["success"]), m["sync_dt_s"]


if __name__ == "__main__":
    name, n, levels = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    jobs = [(name, lv, 5000 + i) for lv in levels for i in range(n)]
    with Pool(workers) as p:
        res = p.map(run_one, jobs, chunksize=1)
    for lv in levels:
        rows = [r for r in res if r[0] == lv]
        ends = Counter(e for r in rows for e in r[1])
        pair_bad = sum(any(e != "rest" for e in r[1]) for r in rows)
        pair_crash = sum(any(e in ("crash", "oob", "tilt", "stall") for e in r[1]) for r in rows)
        lander_bad = sum(e != "rest" for r in rows for e in r[1])
        joint = sum(r[2] for r in rows)
        gaps = [r[3] for r in rows if r[3] == r[3]]
        print(f"level {lv}: n={len(rows)} | pairs with a crash {pair_crash} ({100*pair_crash/len(rows):.1f}%), pairs not both at rest {pair_bad} ({100*pair_bad/len(rows):.1f}%), "
              f"landers not at rest {lander_bad}/{2*len(rows)} ({100*lander_bad/(2*len(rows)):.1f}%), joint success {joint} ({100*joint/len(rows):.0f}%) | ends {dict(ends)} | gap median {np.median(gaps) if gaps else float('nan'):.2f}s", flush=True)
