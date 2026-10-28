"""retrain/episodes.csv, summary.csv, summary.md, chart_success_by_level_after.png, chart_touchdown_speed_hist_after.png from the held-out TEST run of the retrained controller.

    python paper_assets/retrain/scripts/make_after.py <final_checkpoint_name>

Inputs:  paper_assets/retrain/test_new.csv   (run_eval.py --controllers <name> --episodes 300 --seed0 0, the test seeds 0-299)
         paper_assets/before/episodes.csv    (the old results, never changed)
Output rows keep the old controllers' rows exactly as they were in before/ and add the new controller (named `sx_retrained`).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper_assets" / "scripts"))
from make_summary import REASONS, cp_lower_one_sided, summarize, wilson  # noqa: E402
from src.utils import config as C  # noqa: E402

PA = ROOT / "paper_assets"
OUT = PA / "retrain"
names = {f.name for f in font_manager.fontManager.ttflist}
SERIF = next((n for n in ("Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "STIXGeneral", "DejaVu Serif") if n in names), "serif")
plt.rcParams.update({"font.family": "serif", "font.serif": [SERIF], "mathtext.fontset": "stix", "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "axes.edgecolor": "#222", "axes.linewidth": 0.8, "font.size": 11})
# same colours as paper_assets/chart_success_by_level.png; the retrained controller gets its own
LABEL = {"sx_imitation_quiet": "Imitation controller, before", "sx_retrained": "Imitation controller, retrained (after)", "pair_safe_final": "Pair controller (per lander)", "expert_tuned": "Scripted expert"}
COLOR = {"sx_imitation_quiet": "#1f4e8c", "sx_retrained": "#2e7d32", "pair_safe_final": "#b5651d", "expert_tuned": "#5a5a5a"}


def main():
    final = sys.argv[1]
    before = list(csv.DictReader(open(PA / "before" / "episodes.csv", newline="")))
    new = list(csv.DictReader(open(OUT / "test_new.csv", newline="")))
    for r in new:
        r["controller"] = "sx_retrained"
    rows = before + new
    fields = list(before[0].keys())
    with open(OUT / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    order = [c for c in LABEL if any(r["controller"] == c for r in rows)]
    table = []
    for c in order:
        table += summarize([r for r in rows if r["controller"] == c], c)
    cols = list(table[0].keys())
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(table)
    f3 = lambda v: "" if v == "" else f"{float(v):.3f}"
    md = [f"# Retrained single-lander controller (`{final}`) on the held-out test seeds 0-299 (95 % Wilson intervals; 3 decimals; full precision in summary.csv)", "",
          "`sx_retrained` is the new controller; the other rows are the old results copied unchanged from `paper_assets/before/` (same evaluator, same seeds). `pair_safe_final` counts landers (two per episode).", "",
          "| controller | level | n | successes | success rate | Wilson lower | Wilson upper | mean speed (m/s) | median speed (m/s) | mean tilt (rad) | mean pad offset (m) | mean fuel left (kg) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        md.append(f"| {r['controller']} | L{r['level']} | {r['n']} | {r['successes']} | {f3(r['success_rate'])} | {f3(r['wilson_lower'])} | {f3(r['wilson_upper'])} | {f3(r['speed_mean'])} | "
                  f"{f3(r['speed_median'])} | {f3(r['tilt_mean'])} | {f3(r['pad_offset_mean'])} | {f3(r['fuel_remaining_mean_kg'])} |")
    md += ["", "## Failure reasons (counts)", "", "| controller | level | " + " | ".join(REASONS) + " |", "|---|---|" + "---|" * len(REASONS)]
    for r in table:
        md.append(f"| {r['controller']} | L{r['level']} | " + " | ".join(str(r["n_" + x]) for x in REASONS) + " |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    # ---- success chart with before and after bars
    levels = [0, 1, 2, 3]
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=300)
    w = 0.8 / len(order)
    for i, c in enumerate(order):
        rate, lo, hi = [], [], []
        for lv in levels:
            rs = [r for r in rows if r["controller"] == c and int(r["level"]) == lv]
            k, n = sum(int(r["success"]) for r in rs), len(rs)
            l, h = wilson(k, n)
            rate.append(k / n)
            lo.append(max(0.0, k / n - l))
            hi.append(max(0.0, h - k / n))
        x = np.arange(len(levels)) + (i - (len(order) - 1) / 2) * w
        ax.bar(x, rate, w * 0.92, yerr=[lo, hi], color=COLOR[c], capsize=2.5, error_kw={"elinewidth": 0.9, "ecolor": "#111"}, label=LABEL[c])
    ax.set_xticks(np.arange(len(levels)))
    ax.set_xticklabels([f"L{lv}\n(wind scale {C.LEVELS[lv]['wind_power']:g} m/s)" for lv in levels])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Landing success rate")
    ax.set_title("Landing success by difficulty level, before and after retraining (95 % Wilson intervals)", fontsize=11)
    ax.grid(axis="y", color="#ddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "chart_success_by_level_after.png")
    plt.close(fig)
    # ---- touchdown speed histogram of the new controller (old one and the expert as thin references)
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.4), dpi=300)
    top = 30.0
    bins = np.arange(0, top + 0.5, 0.5)
    for ax, lv in zip(axes, levels):
        for c, lw, ls in (("sx_retrained", 1.6, "-"), ("sx_imitation_quiet", 0.9, "-"), ("expert_tuned", 0.9, ":")):
            v = np.array([float(r["touchdown_speed_mps"]) for r in rows if r["controller"] == c and int(r["level"]) == lv and r["touchdown_speed_mps"] != ""])
            ax.hist(np.clip(v, 0, top - 1e-6), bins=bins, histtype="step", linewidth=lw, linestyle=ls, color=COLOR[c], density=True, label=LABEL[c])
        ax.axvline(C.SUCCESS_MAX_SPEED_MPS, color="#c00000", linestyle="--", linewidth=1.1, label=f"success limit ({C.SUCCESS_MAX_SPEED_MPS:g} m/s)")
        ax.set_title(f"L{lv}", fontsize=11)
        ax.set_xlabel("Touchdown speed (m/s)")
        ax.set_xlim(0, top)
        ax.grid(color="#e4e4e4", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Density")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle(f"Touchdown speed per level, retrained controller (speeds above {top:g} m/s are in the last bin)", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(OUT / "chart_touchdown_speed_hist_after.png")
    plt.close(fig)
    print("\n".join(md[:24]))


if __name__ == "__main__":
    main()
