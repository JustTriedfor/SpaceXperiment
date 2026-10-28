"""DAgger rounds: python -m src.sim.dagger_loop  (edit cur / data / rounds below)."""
import subprocess, sys, shutil, os
from pathlib import Path
from src.sim.expert_data import dagger_collect
from src.utils import config as C

here = Path(__file__).resolve().parents[2]
py = sys.executable
cur = "sx_expert_clone"
data = ["expert_sx"]
for rnd in range(1, 7):
    out = f"dagger_{rnd}"
    shutil.rmtree(C.DATA_DIR / "trajectories" / out, ignore_errors=True)
    dagger_collect(cur, 160, out, levels=(0, 0, 1, 1, 2), seed=rnd, log=lambda s: print(s, flush=True))
    data.append(out)
    new = f"sx_dagger_{rnd}"
    cmd = [py, "src/training/finetune.py", "--checkpoint", cur, "--out", new, "--epochs", "30", "--lr", "0.0005", "--batch", "512", "--log-std", "-1.6", "--low-weight", "5"]
    for d in data: cmd += ["--data", d]
    subprocess.run(cmd, check=True, cwd=here)
    cur = new
    print(f"[round {rnd}] trained {cur}", flush=True)
    subprocess.run([py, "src/sim/eval_policy.py", cur, "30"], cwd=here)
    sys.stdout.flush()
