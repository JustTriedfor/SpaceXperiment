"""Job 4 table and the combined results.json (jobs 1-4) with the >3 percentage-point flags.   python results/ablation_2026-10/scripts/make_results.py"""
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, wilson  # noqa: E402

CURRENT = {"trained": [300 / 300, 259 / 300, 161 / 300, 90 / 300], "expert": [1.0, 0.86, 0.54, 0.32]}       # the numbers quoted in the brief (expert L0 was not quoted: 1.0 assumed from the earlier run)
res = {"job1": json.load(open(OUT / "data" / "job12.json"))["job1"], "job2": json.load(open(OUT / "data" / "job12.json"))["job2"], "job3": json.load(open(OUT / "data" / "job3.json")),
       "job3_fine": json.load(open(OUT / "data" / "job3_fine.json")), "job4": {}}
flags = []
pp = lambda a, b: 100 * (a - b)
# ---- job 1 flags
for lv in range(4):
    d = pp(res["job1"][f"L{lv}"]["rate"], CURRENT["expert"][lv])
    if abs(d) > 3:
        flags.append(f"job1 expert L{lv}: {res['job1'][f'L{lv}']['rate']:.3f} vs {CURRENT['expert'][lv]} ({d:+.1f} pp)")
# ---- job 3 flags (against the clean run of the same job)
for k, v in {**res["job3"], **res["job3_fine"]}.items():
    base = res["job3"][f"noise0_L{v['level']}"]["rate"]
    if v["noise_percent"] > 0 and abs(pp(v["rate"], base)) > 3:
        flags.append(f"job3 noise {v['noise_percent']} % L{v['level']}: {v['rate']:.3f} vs {base:.3f} clean ({pp(v['rate'], base):+.1f} pp)")
# ---- job 4
seeds = [s for s in range(1, 6) if (OUT / "data" / "job4" / f"test_abl_s{s}.csv").exists()]
per = {}
for s in seeds:
    rows = list(csv.DictReader(open(OUT / "data" / "job4" / f"test_abl_s{s}.csv")))
    per[s] = [sum(int(r["success"]) for r in rows if int(r["level"]) == lv) for lv in range(4)]
    rr = json.load(open(OUT / "data" / "job4" / "logs" / f"abl_s{s}_result.json"))
    best = next(x for x in rr["rounds"] if x["checkpoint"] == rr["final"])
    res["job4"][f"seed{s}"] = dict(successes=per[s], rates=[k / 300 for k in per[s]], final_checkpoint=rr["final"], validation_mean=best["val"]["mean"], validation_levels=best["val"]["levels"], train_seconds=rr["seconds"])
    for lv in range(4):
        d = pp(per[s][lv] / 300, CURRENT["trained"][lv])
        if abs(d) > 3:
            flags.append(f"job4 seed {s} L{lv}: {per[s][lv]}/300 = {per[s][lv] / 300:.3f} vs current {CURRENT['trained'][lv]:.3f} ({d:+.1f} pp)")
stat = {}
for lv in range(4):
    r = [per[s][lv] / 300 for s in seeds]
    stat[f"L{lv}"] = dict(mean=statistics.mean(r), sd=statistics.stdev(r) if len(r) > 1 else float("nan"), min=min(r), max=max(r), n_seeds=len(r))
    d = pp(stat[f"L{lv}"]["mean"], CURRENT["trained"][lv])
    if abs(d) > 3:
        flags.append(f"job4 mean over seeds L{lv}: {stat[f'L{lv}']['mean']:.3f} vs current {CURRENT['trained'][lv]:.3f} ({d:+.1f} pp)")
res["job4_stats"] = stat
res["flags_over_3pp"] = flags
res["current_numbers_used"] = CURRENT
json.dump(res, open(OUT / "results.json", "w"), indent=1)
md = ["# Job 4: five training seeds (existing recipe), each tested once on seeds 0-299", "",
      "Training seeds 1-5 (the current checkpoint `rt1_final` is training seed 0). Each cell: successes / 300 = rate. SD is the sample standard deviation (n - 1) of the five rates.", "",
      "| training seed | final round (validation mean) | L0 | L1 | L2 | L3 |", "|---|---|---|---|---|---|"]
for s in seeds:
    d = res["job4"][f"seed{s}"]
    md.append(f"| {s} | {d['final_checkpoint'].split('_')[-1] if 'dagger' in d['final_checkpoint'] else 'clone'} ({d['validation_mean']:.3f}) | " + " | ".join(f"{per[s][lv]}/300 = {per[s][lv] / 300:.3f}" for lv in range(4)) + " |")
md.append("| current (seed 0) | dagger_4 (0.638) | " + " | ".join(f"{round(CURRENT['trained'][lv] * 300)}/300 = {CURRENT['trained'][lv]:.3f}" for lv in range(4)) + " |")
md += ["", "| level | mean | SD | min | max | current | mean minus current (pp) |", "|---|---|---|---|---|---|---|"]
for lv in range(4):
    st = stat[f"L{lv}"]
    md.append(f"| L{lv} | {st['mean']:.3f} | {st['sd']:.3f} | {st['min']:.3f} | {st['max']:.3f} | {CURRENT['trained'][lv]:.3f} | {pp(st['mean'], CURRENT['trained'][lv]):+.1f} |")
(OUT / "job4_training_seeds.md").write_text("\n".join(md) + "\n")
(OUT / "flags.md").write_text("# Differences from the current numbers of more than 3 percentage points\n\n" + ("\n".join(f"- {f}" for f in flags) if flags else "None.") + "\n")
print("\n".join(md)); print("\nFLAGS:", *flags, sep="\n")
