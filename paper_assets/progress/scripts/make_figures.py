"""fig3_1_early_versions.png (three real captures side by side) and timeline.png (the milestones of milestones.md on a date axis).

    python paper_assets/progress/scripts/make_figures.py
"""
from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_history import MILESTONES  # noqa: E402

names = {f.name for f in font_manager.fontManager.ttflist}
SERIF = next((n for n in ("Times New Roman", "Times", "Liberation Serif", "STIXGeneral", "DejaVu Serif") if n in names), "serif")
plt.rcParams.update({"font.family": "serif", "font.serif": [SERIF], "mathtext.fontset": "stix", "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})

SHORT = ["first lunar-lander environment", "first PPO training + GIFs", "levels L0-L3, wind, fuel", "control panel + sweeps", "real descent figures", "reward v2, merit, rules stages",
         "network 'brain view'", "imitation tooling, viewer panels", "own 2-D physics (replaces Box2D)", "wind with shear, re-entry spawn", "BC / DAgger experiments", "scripted expert",
         "dual landing, pair, pipeline", "SpaceXEnv on own physics", "landing cam hull, airflow picture"]


def composite():
    items = [("img_01_2026-10-18_lunarlander-prototype-gif-frame.png", "(a) Rocketeer prototype, 18 Oct 2026",
              "Box2D lunar lander, frame 43 of assets/custom_l0_m2.gif (existing recording, unedited)"),
             ("img_02_2026-10-19_rocketeer-box2d-viewer.png", "(b) Rocketeer viewer, 19 Oct 2026",
              "browser viewer with telemetry, phase portrait and nadir cam; run custom_s0_staged8, episode 4428"),
             ("img_03_2026-10-21_spacexperiment-current.png", "(c) SpaceXperiment, 21 Oct 2026",
              "own physics, real-scale Falcon 9 stage, landing cam; imitation network, level 1, seed 122")]
    fig = plt.figure(figsize=(20, 7), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[7, 2.2], hspace=0.03, wspace=0.03, left=0.01, right=0.99, top=0.985, bottom=0.01)
    for k, (f, head, cap) in enumerate(items):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(Image.open(OUT / f))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#999")
        ax = fig.add_subplot(gs[1, k])
        ax.axis("off")
        ax.text(0.5, 0.78, head, ha="center", va="top", fontsize=17, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.45, textwrap.fill(cap, 52), ha="center", va="top", fontsize=13.5, transform=ax.transAxes)
    fig.savefig(OUT / "fig3_1_early_versions.png")
    plt.close(fig)


def timeline():
    t = [datetime.strptime(m[0], "%Y-%m-%d %H:%M") for m in MILESTONES]
    fig, ax = plt.subplots(figsize=(20, 8.2), dpi=100)
    split = datetime(2026, 10, 19, 21, 2)
    ax.axvspan(datetime(2026, 10, 18, 12, 0), split, color="#e8eef7", zorder=0)
    ax.axvspan(split, datetime(2026, 10, 21, 8, 0), color="#f7efe4", zorder=0)
    ax.text(datetime(2026, 10, 18, 12, 30), 6.1, "Rocketeer prototype (Box2D)", fontsize=12, color="#1f4e8c", style="italic")
    ax.text(datetime(2026, 10, 19, 22, 0), 6.1, "SpaceXperiment (own physics)", fontsize=12, color="#8a4b0f", style="italic")
    ax.axhline(0, color="#222", lw=1.2, zorder=1)
    rows = [1.1, -1.1, 2.6, -2.6, 4.1, -4.1]                               # six rows: neighbours in time never share a row, so no two boxes overlap
    for i, (d, s) in enumerate(zip(t, SHORT)):
        y = rows[i % 6]
        ax.plot([d, d], [0, y], color="#777", lw=0.8, zorder=2)
        ax.plot(d, 0, "o", color="#1f4e8c" if d < split else "#b5651d", ms=7, zorder=4)
        label = f"{i + 1}. " + textwrap.fill(s, 22) + "\n" + d.strftime("%d %b %H:%M")
        ax.text(d, y, label, ha="center", va="center", fontsize=9.6, zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#aaa", lw=0.6))
    ax.set_ylim(-5.2, 6.7)
    ax.set_xlim(datetime(2026, 10, 18, 7, 0), datetime(2026, 10, 21, 8, 0))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=(6, 12, 18)))
    ax.set_yticks([])
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Development milestones (dates from file-system times: the project is not under version control)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "timeline.png")
    plt.close(fig)


if __name__ == "__main__":
    composite()
    timeline()
    print("wrote fig3_1_early_versions.png and timeline.png")
