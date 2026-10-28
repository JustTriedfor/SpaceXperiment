"""Fly a dual-lander checkpoint deterministically:  python -m src.sim.eval_dual <checkpoint name> [episodes] [--levels 0,1,2]

Per level: both at rest, joint success (both soft, on their pads and within the landing-together tolerance), the landing-time difference and how
each lander ended.
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from src.sim.dual_env import DualEnv


def main():
    from stable_baselines3 import PPO
    from src.training.weights_io import resolve_checkpoint
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("episodes", type=int, nargs="?", default=30)
    ap.add_argument("--levels", default="0,1,2")
    a = ap.parse_args()
    model = PPO.load(str(resolve_checkpoint(a.checkpoint)), device="cpu")
    for level in (int(x) for x in a.levels.split(",")):
        env = DualEnv(level=level)
        env.set_rules_stage(2)
        both = joint = 0
        dts, ends = [], Counter()
        for ep in range(a.episodes):
            obs, _ = env.reset(seed=1000 + ep)
            done = False
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, _, te, tr, info = env.step(act)
                done = te or tr
            m = info["episode_metrics"]
            both += m["end"] == "rest"
            joint += m["success"]
            if m["sync_dt_s"] == m["sync_dt_s"]:
                dts.append(m["sync_dt_s"])
            ends.update(env.end)
        n = a.episodes
        print(f"level {level}: both at rest {both}/{n}, joint success {joint}/{n} | landing-time gap median {np.median(dts) if dts else float('nan'):.2f} s, "
              f"max {max(dts) if dts else float('nan'):.2f} s | landers: {dict(ends)}", flush=True)


if __name__ == "__main__":
    main()
