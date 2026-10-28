"""Unattended run: finish the expert tuning, apply it, test it, train a pair brain from it and evaluate it.

    python tools/expert_tuning/run_all.py <tuner-log> <tuned.json>

Stages (each logged to logs/pair_safe_run.log; a final summary goes to logs/pair_safe_report.md):
  1 wait for the expert tuner to print its validation line (max 15 min), then read its best constants
  2 back up src/sim/expert.py and write the tuned constants into it
  3 measure the tuned expert on pairs (levels 0-3, 100 pairs each)
  4 run the whole test suite
  5 the pair pipeline (expert demonstrations -> clone -> DAgger rounds -> evaluation) starting from two copies of sx_imitation_quiet, named pair_safe
  6 evaluate pair_safe_final (and the old pair_quiet for comparison) with 100 pairs per level
"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LOG = ROOT / "logs" / "pair_safe_run.log"
LOG.parent.mkdir(exist_ok=True)
NAME = "pair_safe"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, label, timeout=None):
    log(f"START {label}: {' '.join(map(str, cmd))}")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-14:])
    log(f"DONE  {label} (exit {p.returncode}, {time.time() - t0:.0f} s)\n{tail}")
    return p.returncode, out


def main():
    tuner_log, tuned = Path(sys.argv[1]), Path(sys.argv[2])
    log("run_all started")
    # 1 wait for the tuner
    t0 = time.time()
    while time.time() - t0 < 900:
        if tuner_log.exists() and "VALIDATION" in tuner_log.read_text(errors="ignore"):
            break
        time.sleep(15)
    log("tuner: " + (tuner_log.read_text(errors="ignore").strip().splitlines()[-1] if tuner_log.exists() else "no log"))
    params = json.loads(tuned.read_text())
    shutil.copy2(tuned, HERE / "tuned_expert_final.json")
    # 2 apply to expert.py
    ex = ROOT / "src" / "sim" / "expert.py"
    shutil.copy2(ex, HERE / "expert_before_tune.py")
    src = ex.read_text(encoding="utf-8")
    for k, v in params.items():
        new = f"{k} = {float(v):.5g}"
        src, n = re.subn(rf"^{k} = [^\s#]+", new, src, flags=re.M)
        if n != 1:
            log(f"WARNING: could not patch {k} ({n} matches)")
    ex.write_text(src, encoding="utf-8")
    log("expert.py patched with " + json.dumps({k: round(float(v), 4) for k, v in params.items()}))
    # 3 the tuned expert on pairs
    run([sys.executable, str(HERE / "eval_par.py"), "EXPERT", "100", "0,1,2,3", "16"], "expert on pairs", timeout=1800)
    # 4 tests
    rc, _ = run([sys.executable, "-m", "pytest", "-q", "-x"], "full test suite", timeout=1800)
    tests_ok = rc == 0
    # 5 the pair pipeline
    cfg = dict(name=NAME, pair=True, dual=True, from_single="sx_imitation_quiet", steps=["expert", "clone", "dagger", "eval"], seed=0, overwrite=True,
               expert=dict(episodes=500, noise=0.12, levels="0,0,1,1,1,2"),
               clone=dict(epochs=40, lr=1e-3, batch=512, log_std=-1.6, low_weight=5.0),
               dagger=dict(rounds=8, episodes=150, epochs=30, lr=5e-4, batch=512, noise=0.1, levels="0,1,1,1,2", log_std=-1.6, low_weight=5.0),
               eval=dict(episodes=30, levels="0,1"))
    cfg_dir = ROOT / "logs" / "pipeline" / NAME
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    rc, _ = run([sys.executable, "-m", "src.sim.pipeline", "--config", str(cfg_dir / "config.json")], "pair pipeline", timeout=6 * 3600)
    # 6 final evaluation
    rep = []
    for nm in (f"{NAME}_final", "pair_quiet"):
        rc2, out = run([sys.executable, str(HERE / "eval_par.py"), nm, "100", "0,1,2", "16"], f"evaluate {nm}", timeout=3600)
        rep.append((nm, out.strip()))
    lines = ["# pair_safe run report", f"tests passed: {tests_ok}", f"pipeline exit code: {rc}", "", "## evaluation, 100 pairs per level"]
    for nm, out in rep:
        lines += [f"### {nm}", "```", out, "```"]
    (ROOT / "logs" / "pair_safe_report.md").write_text("\n".join(lines), encoding="utf-8")
    log("run_all finished; report in logs/pair_safe_report.md")


if __name__ == "__main__":
    main()
