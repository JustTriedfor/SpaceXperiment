"""chart_success_by_level.png and chart_touchdown_speed_hist.png from episodes.csv (matplotlib, white background, Times-style serif).

    python paper_assets/scripts/make_charts.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper_assets" / "scripts"))
from make_summary import wilson  # noqa: E402
from src.utils import config as C  # noqa: E402

OUT = ROOT / "paper_assets"
names = {f.name for f in font_manager.fontManager.ttflist}
SERIF = next((n for n in ("Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "STIXGeneral", "DejaVu Serif") if n in names), "serif")
plt.rcParams.update({"font.family": "serif", "font.serif": [SERIF], "mathtext.fontset": "stix", "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "axes.edgecolor": "#222", "axes.linewidth": 0.8, "font.size": 11})
LABEL = {"sx_imitation_quiet": "Imitation controller (single lander)", "pair_safe_final": "Pair controller (per lander)", "expert_tuned": "Scripted expert"}
COLOR = {"sx_imitation_quiet": "#1f4e8c", "pair_safe_final": "#b5651d", "expert_tuned": "#5a5a5a"}


def load():
    return list(csv.DictReader(open(OUT / "episodes.csv", newline="")))


def main():
    rows = load()
    ctrls = [c for c in LABEL if any(r["controller"] == c for r in rows)]
    levels = sorted({int(r["level"]) for r in rows})
    # ---- success rate with Wilson error bars
    fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=300)
    w = 0.8 / len(ctrls)
    for i, c in enumerate(ctrls):
        rate, lo, hi = [], [], []
        for lv in levels:
            rs = [r for r in rows if r["controller"] == c and int(r["level"]) == lv]
            k, n = sum(int(r["success"]) for r in rs), len(rs)
            l, h = wilson(k, n)
            rate.append(k / n)
            lo.append(max(0.0, k / n - l))
            hi.append(max(0.0, h - k / n))
        x = np.arange(len(levels)) + (i - (len(ctrls) - 1) / 2) * w
        ax.bar(x, rate, w * 0.92, yerr=[lo, hi], color=COLOR[c], capsize=2.5, error_kw={"elinewidth": 0.9, "ecolor": "#111"}, label=LABEL[c])
    ax.set_xticks(np.arange(len(levels)))
    ax.set_xticklabels([f"L{lv}\n(wind scale {C.LEVELS[lv]['wind_power']:g} m/s)" for lv in levels])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Landing success rate")
    ax.set_title("Landing success by difficulty level (95 % Wilson intervals)", fontsize=11)
    ax.grid(axis="y", color="#ddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "chart_success_by_level.png")
    plt.close(fig)
    # ---- touchdown speed histograms
    fig, axes = plt.subplots(1, len(levels), figsize=(3.2 * len(levels), 3.4), dpi=300, sharey=False)
    axes = np.atleast_1d(axes)
    top = 30.0
    bins = np.arange(0, top + 0.5, 0.5)
    for ax, lv in zip(axes, levels):
        for c in ctrls:
            v = np.array([float(r["touchdown_speed_mps"]) for r in rows if r["controller"] == c and int(r["level"]) == lv and r["touchdown_speed_mps"] != ""])
            ax.hist(np.clip(v, 0, top - 1e-6), bins=bins, histtype="step", linewidth=1.3, color=COLOR[c], density=True, label=LABEL[c])
        ax.axvline(C.SUCCESS_MAX_SPEED_MPS, color="#c00000", linestyle="--", linewidth=1.1, label=f"success limit ({C.SUCCESS_MAX_SPEED_MPS:g} m/s)")
        ax.set_title(f"L{lv}", fontsize=11)
        ax.set_xlabel("Touchdown speed (m/s)")
        ax.set_xlim(0, top)
        ax.grid(color="#e4e4e4", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Density")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle(f"Touchdown speed per level (speeds above {top:g} m/s are in the last bin)", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(OUT / "chart_touchdown_speed_hist.png")
    plt.close(fig)
    print("font:", SERIF, "| wrote chart_success_by_level.png and chart_touchdown_speed_hist.png")


if __name__ == "__main__":
    main()
