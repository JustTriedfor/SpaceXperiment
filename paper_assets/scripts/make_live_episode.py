"""Records ONE representative landing of the trained controller (sx_imitation_quiet) as a viewer episode, for the UI figures.

    python paper_assets/scripts/make_live_episode.py [--level 1]

The representative episode: among the level's successful episodes in episodes.csv, the one whose touchdown speed and pad offset are closest to the medians of the
successes (so it is typical, not the best). It is flown again with the same seed and the same deterministic policy, with the trajectory recorded and the network's
activations attached (what the training run streams to the viewer), and written to logs/zz_paper_view/live/ (a throw-away run folder; capture_ui.py deletes it).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.sim.env import SpaceXEnv
from src.training.brain import episode_payload
from src.training.weights_io import resolve_checkpoint

OUT = ROOT / "logs" / "zz_paper_view" / "live"


def pick(level: int, controller: str = "sx_imitation_quiet") -> int:
    rows = [r for r in csv.DictReader(open(ROOT / "paper_assets" / "episodes.csv")) if r["controller"] == controller and int(r["level"]) == level and r["success"] == "1"]
    ms = statistics.median(float(r["touchdown_speed_mps"]) for r in rows)
    mo = statistics.median(float(r["pad_offset_m"]) for r in rows)
    sd = statistics.pstdev(float(r["touchdown_speed_mps"]) for r in rows) or 1.0
    so = statistics.pstdev(float(r["pad_offset_m"]) for r in rows) or 1.0
    best = min(rows, key=lambda r: ((float(r["touchdown_speed_mps"]) - ms) / sd) ** 2 + ((float(r["pad_offset_m"]) - mo) / so) ** 2)
    return int(best["seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    from stable_baselines3 import PPO
    model = PPO.load(str(resolve_checkpoint("sx_imitation_quiet")), device="cpu")
    seed = a.seed if a.seed is not None else pick(a.level)
    env = SpaceXEnv(level=a.level, record=True)
    env.set_rules_stage(2)
    obs, _ = env.reset(seed=seed)
    O, A, done = [], [], False
    while not done:
        act = model.predict(obs, deterministic=True)[0]
        O.append(obs.copy())
        A.append(np.asarray(act, dtype=np.float32))
        obs, _, te, tr, info = env.step(act)
        done = te or tr
    traj = info["trajectory"]
    traj["act"] = episode_payload(model.policy, np.array(O), np.array(A))
    traj.update(id=1, env=0, timesteps=1000)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT.parent / "episodes.csv").write_text("timesteps\n")
    (OUT / "ep_00000001_e0.json").write_text(json.dumps(traj, separators=(",", ":")))
    r = traj["result"]
    print(f"level {a.level}, seed {seed}: end {r['end']}, success {r['success']}, speed {r['speed']} m/s, offset {r['dist']} m, {len(traj['frames'])} frames")


if __name__ == "__main__":
    main()
