"""Fly a checkpoint deterministically and report how it lands (single lander or Dual land, detected from the checkpoint).

    python -m src.sim.evaluate <checkpoint> [--episodes 30] [--levels 0,1,2] [--json out.json]

Per level: soft-landing successes, outcomes, median touchdown speed / miss / propellant; for Dual land also both-at-rest, joint successes
(both soft within the landing-together tolerance) and the landing-time gap. Used by the Pipeline and Checkpoints tabs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.utils import config as C


def evaluate(checkpoint, levels=(0, 1, 2), episodes: int = 30, seed0: int = 1000, log=None) -> dict:
    """-> dict(checkpoint, kind, episodes, levels=[{level, n, success, rest, ends, speed, miss, fuel, ...}], score)  (score = mean success rate)"""
    from stable_baselines3 import PPO
    from src.training.pair import load_pair_models, resolve_pair
    from src.training.weights_io import resolve_checkpoint
    pair = resolve_pair(checkpoint)
    if pair is not None:                                            # two networks, one per lander
        from src.sim.dual_env import DualEnv as Env, agent_view
        mA, mB = load_pair_models(pair[0])
        dual, kind, label = True, "pair", pair[0]

        def act_for(obs):
            return np.concatenate([mA.predict(agent_view(obs, 0), deterministic=True)[0], mB.predict(agent_view(obs, 1), deterministic=True)[0]])
    else:
        src = resolve_checkpoint(checkpoint)
        model = PPO.load(str(src), device="cpu")
        dual = int(model.observation_space.shape[0]) == C.DUAL_OBS_DIM
        kind, label = ("dual" if dual else "single"), src.stem
        if dual:
            from src.sim.dual_env import DualEnv as Env
        else:
            from src.sim.env import SpaceXEnv as Env

        def act_for(obs):
            return model.predict(obs, deterministic=True)[0]
    out = []
    for level in levels:
        env = Env(level=int(level))
        env.set_rules_stage(2)
        ends, speeds, miss, fuel, gaps = Counter(), [], [], [], []
        success = rest = 0
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed0 + ep)
            done = False
            while not done:
                obs, _, te, tr, info = env.step(act_for(obs))
                done = te or tr
            m = info["episode_metrics"]
            success += bool(m["success"])
            rest += m["end"] == "rest"
            if dual:
                ends.update(env.end)
                if m["sync_dt_s"] == m["sync_dt_s"]:
                    gaps.append(m["sync_dt_s"])
            else:
                ends[m["end"]] += 1
            if m["touchdown"]:
                speeds.append(m["speed_mps"])
                miss.append(m["dist_m"])
            fuel.append(m["fuel_used"])
        row = dict(level=int(level), n=episodes, success=success, rest=rest, ends=dict(ends),
                   speed=float(np.median(speeds)) if speeds else None, miss=float(np.median(miss)) if miss else None, fuel=float(np.median(fuel)))
        if dual:
            row.update(gap_median=float(np.median(gaps)) if gaps else None, gap_max=float(max(gaps)) if gaps else None)
        out.append(row)
        if log:
            log(describe(row, dual))
    return dict(checkpoint=label, kind=kind, episodes=episodes, levels=out,
                score=float(np.mean([r["success"] / max(r["n"], 1) for r in out])) if out else 0.0)


def describe(r: dict, dual: bool) -> str:
    f = lambda v, d=1: "n/a" if v is None else f"{v:.{d}f}"
    if dual:
        return (f"level {r['level']}: joint success {r['success']}/{r['n']}, both at rest {r['rest']}/{r['n']} | gap median {f(r.get('gap_median'), 2)} s, max {f(r.get('gap_max'), 2)} s"
                f" | speed {f(r['speed'])} m/s, miss {f(r['miss'], 0)} m, propellant {f(r['fuel'], 0)} kg (both) | landers {r['ends']}")
    return (f"level {r['level']}: success {r['success']}/{r['n']} | ends {r['ends']} | median touchdown speed {f(r['speed'])} m/s, miss {f(r['miss'], 0)} m, "
            f"propellant {f(r['fuel'], 0)} kg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--levels", default="0,1,2")
    ap.add_argument("--json", default=None, help="also write the result here")
    a = ap.parse_args()
    levels = [int(x) for x in a.levels.split(",")]
    res = evaluate(a.checkpoint, levels, a.episodes, log=lambda s: print(s, flush=True))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
