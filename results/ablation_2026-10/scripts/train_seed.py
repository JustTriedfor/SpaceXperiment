"""Job 4, one training seed: the existing retraining recipe (paper_assets/retrain/scripts/retrain_single.py, config attempt1.json = all defaults: start from sx_imitation_quiet, 800 expert
demonstrations, behaviour cloning, 8 DAgger rounds on levels 0,1,1,2,2,3,3, final network = the round with the best mean success on the VALIDATION seeds 100000-100199) with a different
TRAINING seed, then ONE test on seeds 0-299 at L0-L3.

    python results/ablation_2026-10/scripts/train_seed.py --seed 1

What the training seed changes: the environment seeds of the expert demonstrations and of every DAgger round (offset by seed * 10,000,000, so no two training seeds share an episode and none
touches the test seeds 0-299 or the validation seeds 100000-100199), the random choice of levels, the demonstration and DAgger noise... (each episode's noise is seeded by its episode seed) and the
shuffling inside the fitting step. The starting network is the same existing checkpoint for every seed (that is the existing recipe).
Outputs: data/job4/abl_s<seed>*.log / _result.json (training log with the validation scores of every round), data/job4/test_abl_s<seed>.csv (test episodes).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "paper_assets" / "retrain" / "scripts"))
import retrain_single as RS  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    assert 1 <= a.seed <= 50
    s = a.seed
    job = OUT / "data" / "job4"
    job.mkdir(parents=True, exist_ok=True)
    RS.EXPERT_SEED0 = 1_000_000 + s * 10_000_000
    RS.DAGGER_SEED0 = RS.EXPERT_SEED0 + 1_000_000
    RS.RETRAIN = job                                              # logs and validation CSVs go into results/ablation_2026-10/data/job4
    cfg_path = job / f"config_abl_s{s}.json"
    cfg_path.write_text(json.dumps({"seed": s}))
    sys.argv = ["retrain_single.py", "--name", f"abl_s{s}", "--config", str(cfg_path)]
    RS.main()
    test = job / f"test_abl_s{s}.csv"
    p = subprocess.run([sys.executable, str(ROOT / "paper_assets" / "scripts" / "run_eval.py"), "--controllers", f"abl_s{s}_final", "--levels", "0,1,2,3", "--episodes", "300", "--seed0", "0",
                        "--workers", "12", "--out", str(test)], capture_output=True, text=True, cwd=ROOT)
    print(p.stdout[-300:], p.stderr[-500:])



if __name__ == "__main__":      # required: the worker processes (Windows spawn) import this file again
    main()
