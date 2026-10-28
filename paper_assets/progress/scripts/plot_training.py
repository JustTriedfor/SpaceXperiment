"""Learning curves from the real training logs kept on disk (white background, serif font, no smoothing: one point per fixed-width bin of training steps,
bins are not overlapping and no moving average is applied; the bin's episode count is shown in the top axis labels of the README table).

    python paper_assets/progress/scripts/plot_training.py

Each run folder's episodes.csv has one row per finished training episode (timesteps = step at which it ended). levels.csv / stages.csv give the steps at which the
curriculum moved (dashed vertical lines).
Inputs (read only):
  train_custom_s0_m2.png        C:/Users/firdo/rocketeer/logs/custom_s0_m2             Rocketeer (Box2D), 18 Oct 2026, 13:49
  train_custom_s0_staged8.png   C:/Users/firdo/rocketeer/logs/custom_s0_staged8        Rocketeer, 18 Oct 2026, 20:24
  train_custom_s0_quiet.png     C:/Users/firdo/rocketeer/SpaceXperiment/logs/custom_s0_quiet   SpaceXperiment, 20 Oct 2026, 08:49
  train_custom_s0_pair.png      C:/Users/firdo/rocketeer/SpaceXperiment/logs/custom_s0_pair    SpaceXperiment, 21 Oct 2026 (the run was still writing when read)
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

OUT = Path(__file__).resolve().parents[1]
RUNS = [("custom_s0_m2", Path("C:/Users/firdo/rocketeer/logs/custom_s0_m2"), "Rocketeer prototype (Box2D lunar lander), level 0"),
        ("custom_s0_staged8", Path("C:/Users/firdo/rocketeer/logs/custom_s0_staged8"), "Rocketeer prototype, staged rules, 8 environments"),
        ("custom_s0_quiet", Path("C:/Users/firdo/rocketeer/SpaceXperiment/logs/custom_s0_quiet"), "SpaceXperiment (own physics), PPO fine-tuning of the imitation network"),
        ("custom_s0_pair", Path("C:/Users/firdo/rocketeer/SpaceXperiment/logs/custom_s0_pair"), "SpaceXperiment, pair mode (two networks)")]
names = {f.name for f in font_manager.fontManager.ttflist}
SERIF = next((n for n in ("Times New Roman", "Times", "Liberation Serif", "STIXGeneral", "DejaVu Serif") if n in names), "serif")
plt.rcParams.update({"font.family": "serif", "font.serif": [SERIF], "mathtext.fontset": "stix", "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "font.size": 10})


def nice(x):
    e = 10 ** math.floor(math.log10(x))
    return min((m * e for m in (1, 2, 2.5, 5, 10) if m * e >= x), default=x)


def main():
    lines = ["| run | file | episodes | last step | bin width (steps) | file modified |", "|---|---|---|---|---|---|"]
    for name, d, title in RUNS:
        df = pd.read_csv(d / "episodes.csv", usecols=lambda c: c in ("timesteps", "level", "success", "soft", "touchdown", "speed_mps", "dist_m", "end"))
        width = nice(df.timesteps.max() / 40)
        df["bin"] = (df.timesteps // width).astype(int)
        g = df.groupby("bin")
        x = (g.size().index.values + 0.5) * width
        n = g.size().values
        rate = g.success.mean().values
        soft = g.soft.mean().values if "soft" in df else None
        touch = g.touchdown.mean().values
        med = df[df.touchdown].groupby("bin").speed_mps.median().reindex(g.size().index).values
        fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.5), dpi=200)
        ax[0].plot(x, rate, "o-", ms=3.2, lw=1, color="#1f4e8c", label="success (strict rules of that period)")
        if soft is not None:
            ax[0].plot(x, soft, "s-", ms=2.6, lw=0.8, color="#b5651d", label="soft landing (speed limit only)")
        ax[0].plot(x, touch, "-", lw=0.8, color="#888", label="reached the ground")
        ax[0].set_ylim(-0.02, 1.02)
        ax[0].set_ylabel("share of episodes in the bin")
        ax[1].plot(x, med, "o-", ms=3.2, lw=1, color="#7a1f1f")
        ax[1].set_ylabel("median touchdown speed (m/s)")
        if np.nanmax(med) > 30:
            ax[1].set_yscale("symlog", linthresh=5)       # long tail of fast crashes: linear up to 5 m/s, logarithmic above
            ax[1].set_yticks([0, 1, 2, 3, 5, 10, 20, 50, 100])
            ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        for a in ax:
            a.set_xlabel("training step (environment steps)")
            a.grid(color="#e4e4e4", lw=0.5)
            a.set_axisbelow(True)
            for f, col in (("levels.csv", "#2e7d32"), ("stages.csv", "#6a1b9a")):
                fp = d / f
                if fp.exists():
                    cdf = pd.read_csv(fp)
                    for t in cdf.timesteps[1:]:
                        a.axvline(t, color=col, ls="--", lw=0.7, alpha=0.7)
        ax[0].legend(frameon=False, fontsize=7.5, loc="center right")
        fig.suptitle(f"{name}: {title}\n({len(df)} episodes, bins of {width:g} steps, no smoothing; dashed = curriculum level (green) / rules stage (purple) changes)", fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(OUT / f"train_{name}.png")
        plt.close(fig)
        lines.append(f"| {name} | {d.as_posix()}/episodes.csv | {len(df)} | {int(df.timesteps.max())} | {width:g} | {time.strftime('%Y-%m-%d %H:%M', time.localtime((d / 'episodes.csv').stat().st_mtime))} |")
        print(name, len(df), "bins", len(x), "min/median/max episodes per bin", n.min(), int(np.median(n)), n.max())
    (OUT / "training_runs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
