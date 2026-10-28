"""summary.csv and summary.md from episodes.csv (nothing is rounded in the CSV; the markdown shows 3 decimals).

    python paper_assets/scripts/make_summary.py

Rows: one per (controller, level). For pair_safe_final, n counts LANDERS (two per episode); an extra controller 'pair_safe_final (joint)' counts EPISODES in which both
landers succeeded and landed within DUAL_SYNC_TOL_S of each other.
Means of touchdown speed, tilt and pad offset are over the episodes that touched down; the fuel mean is over all episodes.
Wilson: 95 % two-sided score interval. CP lower: Clopper-Pearson ONE-sided 95 % lower bound on the success probability (exact binomial; for no failures it is 0.05**(1/n)).
"""
from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper_assets"
Z = 1.959963984540054
REASONS = ["none", "too_fast", "tilted", "off_pad", "out_of_fuel", "crashed_other"]


def wilson(k, n, z=Z):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cp_lower_one_sided(k, n, alpha=0.05):
    from scipy.stats import beta
    return 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))


def need_clean(p0, alpha=0.05):
    """clean trials needed for a one-sided 95 % bound >= p0: exact binomial (alpha = p0**n) and the rule of three (3 / (1 - p0))."""
    return math.ceil(math.log(alpha) / math.log(p0)), math.ceil(3.0 / (1.0 - p0))


def load():
    return list(csv.DictReader(open(OUT / "episodes.csv", newline="")))


def num(v):
    return float(v) if v not in ("", None) else None


def summarize(rows, name):
    out = []
    for lv in sorted({int(r["level"]) for r in rows}):
        rs = [r for r in rows if int(r["level"]) == lv]
        n = len(rs)
        k = sum(int(r["success"]) for r in rs)
        lo, hi = wilson(k, n)
        spd = [num(r["touchdown_speed_mps"]) for r in rs if r["touchdown_speed_mps"] != ""]
        tlt = [num(r["tilt_rad"]) for r in rs if r["tilt_rad"] != ""]
        off = [num(r["pad_offset_m"]) for r in rs if r["pad_offset_m"] != ""]
        fuel = [num(r["fuel_remaining_kg"]) for r in rs]
        row = dict(controller=name, level=lv, n=n, successes=k, success_rate=k / n, wilson_lower=lo, wilson_upper=hi, cp_one_sided_lower=cp_lower_one_sided(k, n),
                   speed_mean=statistics.mean(spd) if spd else "", speed_median=statistics.median(spd) if spd else "", tilt_mean=statistics.mean(tlt) if tlt else "",
                   pad_offset_mean=statistics.mean(off) if off else "", fuel_remaining_mean_kg=statistics.mean(fuel))
        for reason in REASONS:
            row["n_" + reason] = sum(1 for r in rs if r["failure_reason"] == reason)
        out.append(row)
    return out


def main():
    rows = load()
    table = []
    for c in dict.fromkeys(r["controller"] for r in rows):
        table += summarize([r for r in rows if r["controller"] == c], c)
    pair = [r for r in rows if r["controller"] == "pair_safe_final"]
    joint = []                                                   # one row per episode: both landers
    for lv in sorted({int(r["level"]) for r in pair}):
        for s in sorted({int(r["seed"]) for r in pair if int(r["level"]) == lv}):
            a = next(r for r in pair if int(r["level"]) == lv and int(r["seed"]) == s and r["lander"] == "A")
            joint.append(dict(a, success=a["joint_success"], failure_reason="none" if a["joint_success"] == "1" else "joint_not_met"))
    j = summarize(joint, "pair_safe_final (joint)") if joint else []
    for r in j:                                                  # the joint rows count episodes; the failure counts are not split, only the total
        for reason in REASONS[1:]:
            r["n_" + reason] = ""
    table += j
    cols = list(table[0].keys())
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(table)

    f3 = lambda v: "" if v == "" else f"{float(v):.3f}"
    md = ["# Summary (95 % Wilson intervals; 3 decimals; full precision in summary.csv)", "",
          "n counts episodes, except `pair_safe_final` where it counts landers (two per episode). `pair_safe_final (joint)` counts episodes in which both landers succeeded and landed within the sync tolerance.", "",
          "| controller | level | n | successes | success rate | Wilson lower | Wilson upper | mean speed (m/s) | median speed (m/s) | mean tilt (rad) | mean pad offset (m) | mean fuel left (kg) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        md.append(f"| {r['controller']} | L{r['level']} | {r['n']} | {r['successes']} | {f3(r['success_rate'])} | {f3(r['wilson_lower'])} | {f3(r['wilson_upper'])} | {f3(r['speed_mean'])} | {f3(r['speed_median'])} | {f3(r['tilt_mean'])} | {f3(r['pad_offset_mean'])} | {f3(r['fuel_remaining_mean_kg'])} |")
    md += ["", "## Failure reasons (counts)", "", "| controller | level | " + " | ".join(REASONS) + " |", "|---|---|" + "---|" * len(REASONS)]
    for r in table:
        if r["controller"].endswith("(joint)"):
            continue
        md.append(f"| {r['controller']} | L{r['level']} | " + " | ".join(str(r["n_" + x]) for x in REASONS) + " |")

    n99e, n99r = need_clean(0.99)
    n996e, n996r = need_clean(0.9963)
    md += ["", "## How many clean trials prove a reliability?", "",
           "With every trial a success, the one-sided 95 % lower bound on the success probability p is 0.05^(1/n) (exact binomial), or about 1 - 3/n (rule of three).", "",
           "| claim | exact binomial: clean trials needed | rule of three: clean trials needed |", "|---|---|---|",
           f"| p >= 0.99 | {n99e} | {n99r} |", f"| p >= 0.9963 | {n996e} | {n996r} |", "",
           "What was achieved (one-sided 95 % Clopper-Pearson lower bound, `clean` = no failure in n trials):", "",
           "| controller | level | n | failures | clean | one-sided 95 % lower bound on p | supports p >= 0.99 | supports p >= 0.9963 |", "|---|---|---|---|---|---|---|---|"]
    for r in table:
        fl = r["n"] - r["successes"]
        md.append(f"| {r['controller']} | L{r['level']} | {r['n']} | {fl} | {'yes' if fl == 0 else 'no'} | {f3(r['cp_one_sided_lower'])} | {'yes' if r['cp_one_sided_lower'] >= 0.99 else 'no'} | {'yes' if r['cp_one_sided_lower'] >= 0.9963 else 'no'} |")
    md += ["", "Notes: the two landers of a pair share one wind, one start path and one sync reward, so their outcomes are not independent; the per-lander n for `pair_safe_final` therefore overstates the "
           "information (the joint rows, n = episodes, are the honest count). Each episode of the other controllers is an independent draw (seeds 0-299)."]
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
