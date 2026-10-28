"""Learning-rate sweep: one independent agent per environment, learning rate growing exponentially.

    python src/training/sweep.py --reward custom --n-envs 8 --lr-growth 1.5 [--seed N --tag _x ...]

Agent i gets its own policy, its own environment and learning rate  base_lr * growth**(i // copies)  (base_lr is
PPO learning_rate from the config / app). Every agent runs its own curriculum. Logs go to
logs/<name>/agent_<i>/, live episodes to logs/<name>/live/ (env index = agent index), so the web
app shows the agents side by side. Normal training (train.py) is unchanged: all of its environments
feed one shared policy, which can only have one learning rate.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["custom", "default"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=C.TOTAL_TIMESTEPS, help="per agent")
    ap.add_argument("--n-envs", type=int, default=C.N_ENVS, help="number of agents")
    ap.add_argument("--start-level", type=int, default=0)
    ap.add_argument("--stop-at-level", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lr-growth", type=float, default=1.5)
    ap.add_argument("--copies", type=int, default=1, help="agents per learning rate (different seeds)")
    args = ap.parse_args()

    if args.n_envs % args.copies:
        sys.exit(f"--n-envs {args.n_envs} must be a multiple of --copies {args.copies}")
    n_groups = args.n_envs // args.copies
    group_lrs = C.agent_lrs(C.PPO_KWARGS["learning_rate"], args.lr_growth, n_groups)
    groups = [i // args.copies for i in range(args.n_envs)]      # copies of one rate sit next to each other
    lrs = [group_lrs[g] for g in groups]
    if max(lrs) > C.MAX_LEARNING_RATE:
        sys.exit(f"largest learning rate {max(lrs):.3g} exceeds {C.MAX_LEARNING_RATE}")
    name = f"{args.reward}_s{args.seed}{args.tag}_sweep"
    run_dir = C.LOG_DIR / name
    shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "live").mkdir(parents=True)
    C.MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (C.MODEL_CONFIG_DIR / f"{name}.json").write_text(json.dumps(
        dict(vars(args), name=name, sweep=dict(lrs=lrs, groups=groups, copies=args.copies, growth=args.lr_growth), overrides=C.OVERRIDES), indent=2))

    procs = []
    try:
        for i, lr in enumerate(lrs):
            cmd = [sys.executable, str(Path(__file__).with_name("train.py")),
                   "--reward", args.reward, "--seed", str(args.seed + i), "--total-steps", str(args.total_steps),
                   "--n-envs", "1", "--start-level", str(args.start_level), "--tag", f"{args.tag}_a{i}",
                   "--lr", repr(lr), "--run-dir", str(run_dir / f"agent_{i}"),
                   "--live-dir", str(run_dir / "live"), "--live-env", str(i)]
            if args.stop_at_level is not None:
                cmd += ["--stop-at-level", str(args.stop_at_level)]
            out = open(run_dir / f"agent_{i}.out", "w")
            procs.append(subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT))
            print(f"agent {i}: group {groups[i]} lr={lr:.3g}", flush=True)
            time.sleep(0.5)                                   # stagger start-up
        for p in procs:
            p.wait()
        codes = [p.returncode for p in procs]
        print(f"sweep done, exit codes {codes}", flush=True)
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()


if __name__ == "__main__":
    main()
