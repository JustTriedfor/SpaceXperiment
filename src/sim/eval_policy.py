"""Fly a saved policy deterministically and report how it lands.   python src/sim/eval_policy.py <checkpoint name> [episodes]"""
import sys
from collections import Counter
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)

import numpy as np
from stable_baselines3 import PPO

from src.sim.env import SpaceXEnv

name = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
model = PPO.load(f"{ROOT}/models/checkpoints/{name}.zip", device="cpu")
for level in (0, 1, 2):
    env = SpaceXEnv(level=level)
    env.set_rules_stage(2)
    ends, speeds, miss, fuel, succ = [], [], [], [], 0
    for ep in range(n):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
        m = info["episode_metrics"]
        ends.append(m["end"])
        if m["touchdown"]:
            speeds.append(m["speed_mps"])
            miss.append(m["dist_m"])
        fuel.append(m["fuel_used"])
        succ += m["success"]
    print(f"level {level}: {dict(Counter(ends))} success {succ}/{n} | median touchdown speed {np.median(speeds) if speeds else float('nan'):.1f} m/s, "
          f"miss {np.median(miss) if miss else float('nan'):.0f} m, fuel {np.median(fuel):.0f} kg", flush=True)
