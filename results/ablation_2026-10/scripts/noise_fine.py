"""Job 3, supplementary: the same noise test at smaller noise levels (0.05, 0.1, 0.25, 0.5 %) to locate where the controller breaks.  Not requested in the brief; added because 1 % already gives 0 successes.

    python results/ablation_2026-10/scripts/noise_fine.py [--workers 12]
Same spans (data/obs_ranges.json), same noise draws and same seeds 0-299 as noise_eval.py.
"""
import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise_eval as N  # noqa: E402
from common import OUT, wilson  # noqa: E402

PCTS = (0.05, 0.1, 0.25, 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    span = json.load(open(OUT / "data" / "obs_ranges.json"))["span"]
    with Pool(a.workers, initializer=N._init) as pool:
        rows = pool.map(N.eval_job, [(p, lv, s, span) for p in PCTS for lv in N.LEVELS for s in range(300)], chunksize=4)
    with open(OUT / "data" / "job3_fine_noise_episodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["noise_percent", "level", "seed", "success", "touchdown_speed_mps", "pad_offset_m"])
        w.writerows(sorted(rows))
    res, md = {}, ["# Job 3 (supplementary): smaller noise levels", "", "Same method as job 3, noise 0.05 to 0.5 % of each observation's range (successes / 300 = rate [95 % Wilson interval]).", "",
                   "| noise | L0 | L1 | L2 | L3 |", "|---|---|---|---|---|"]
    for p in PCTS:
        cells = []
        for lv in N.LEVELS:
            k = sum(r[3] for r in rows if r[0] == p and r[1] == lv)
            l, h = wilson(k, 300)
            res[f"noise{p}_L{lv}"] = dict(noise_percent=p, level=lv, n=300, successes=k, rate=k / 300, wilson_lower=l, wilson_upper=h)
            cells.append(f"{k}/300 = {k / 300:.3f} [{l:.3f}, {h:.3f}]")
        md.append(f"| {p} % | " + " | ".join(cells) + " |")
    (OUT / "job3_sensor_noise_fine.md").write_text("\n".join(md) + "\n")
    json.dump(res, open(OUT / "data" / "job3_fine.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
