"""Jobs 1 and 2 from data/job1_job2_episodes.csv (run_eval.py output: expert_tuned and rt1_final, seeds 0-299, L0-L3).   python results/ablation_2026-10/scripts/analyze_job12.py"""
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CHECKPOINT, OUT, wilson  # noqa: E402

rows = list(csv.DictReader(open(OUT / "data" / "job1_job2_episodes.csv")))
NAMES = {"expert_tuned": "scripted expert", CHECKPOINT: "trained controller"}
TYPES = ["too_fast", "off_pad", "tilted", "out_of_fuel", "crashed_other"]
res = {"job1": {}, "job2": {}}
md1 = ["# Job 1: scripted expert, seeds 0-299, 300 episodes per level", "", "| level | successes / 300 | success rate | 95 % Wilson interval |", "|---|---|---|---|"]
for lv in range(4):
    rs = [r for r in rows if r["controller"] == "expert_tuned" and int(r["level"]) == lv]
    k = sum(int(r["success"]) for r in rs)
    lo, hi = wilson(k, len(rs))
    res["job1"][f"L{lv}"] = dict(n=len(rs), successes=k, rate=k / len(rs), wilson_lower=lo, wilson_upper=hi)
    md1.append(f"| L{lv} | {k} / {len(rs)} | {k / len(rs):.3f} | [{lo:.3f}, {hi:.3f}] |")
(OUT / "job1_expert_baseline.md").write_text("\n".join(md1) + "\n")
md2 = ["# Job 2: failure modes (seeds 0-299)", "",
       "One label per failed episode, first match of: out_of_fuel, too_fast (first-contact speed > 3 m/s), tilted (> 0.15 rad), off_pad (> 10 m), otherwise other (no touchdown, fall over, failed settle). "
       "The columns add up to the failures; the last column checks it.", "",
       "| controller | level | successes | failures | too fast | off pad | tilt too large | out of fuel | other | sum of types | sum = failures |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for c in ("expert_tuned", CHECKPOINT):
    for lv in range(4):
        rs = [r for r in rows if r["controller"] == c and int(r["level"]) == lv]
        f = [r for r in rs if r["success"] != "1"]
        cnt = collections.Counter(r["failure_reason"] for r in f)
        tot = sum(cnt[t] for t in TYPES)
        res["job2"][f"{c}_L{lv}"] = dict(controller=NAMES[c], n=len(rs), successes=len(rs) - len(f), failures=len(f), **{t: cnt[t] for t in TYPES}, sum_ok=tot == len(f))
        md2.append(f"| {NAMES[c]} | L{lv} | {len(rs) - len(f)} | {len(f)} | {cnt['too_fast']} | {cnt['off_pad']} | {cnt['tilted']} | {cnt['out_of_fuel']} | {cnt['crashed_other']} | {tot} | {'yes' if tot == len(f) else 'NO'} |")
md2 += ["", "Every rule an episode broke (an episode can break several, so these do not add up to the failures; this shows how often each rule is involved):", "",
        "| controller | level | failures | broke too_fast | broke off_pad | broke tilt | broke out_of_fuel |", "|---|---|---|---|---|---|---|"]
for c in ("expert_tuned", CHECKPOINT):
    for lv in range(4):
        f = [r for r in rows if r["controller"] == c and int(r["level"]) == lv and r["success"] != "1"]
        n = lambda k: sum(1 for r in f if k in r["failure_flags"].split("|"))
        md2.append(f"| {NAMES[c]} | L{lv} | {len(f)} | {n('too_fast')} | {n('off_pad')} | {n('tilted')} | {n('out_of_fuel')} |")
l3 = res["job2"][f"{CHECKPOINT}_L3"]
md2 += ["", f"Check against the paper's numbers (trained controller, L3): {l3['successes']} successes, {l3['failures']} failures, {l3['too_fast']} too fast, {l3['off_pad']} off pad: "
        + ("matches 90 / 210 / 153 / 57." if (l3["successes"], l3["failures"], l3["too_fast"], l3["off_pad"]) == (90, 210, 153, 57) else "DIFFERS from 90 / 210 / 153 / 57.")]
(OUT / "job2_failure_modes.md").write_text("\n".join(md2) + "\n")
json.dump(res, open(OUT / "data" / "job12.json", "w"), indent=1)
print("\n".join(md1)); print("\n".join(md2))
