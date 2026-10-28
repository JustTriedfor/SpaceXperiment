"""Figures 5, 6 and 7 of the REBOUND paper (matplotlib, white background, Arial, PNG 300 dpi + SVG, 16 cm wide, no title inside the image).

    python paper_assets/figures/make_figs_5_6_7.py

fig5_landing_scatter       paper_assets/retrain/episodes.csv, controller `sx_retrained` (= checkpoint rt1_final), the 1,200 test episodes (seeds 0-299, L0-L3)
fig6_descent_phase_portrait  one re-run episode of rt1_final: level 1, seed 122, deterministic policy, nothing changed in the simulator
fig7_emissions_vs_p        the rebuilt Monte Carlo of paper_assets/montecarlo/emissions_mc.py (200,000 draws, N uniform 15-40, one seed) and the Wilson intervals of retrain/summary.csv
No number is changed or re-estimated. Checks (Figure 5 success counts, Figure 7 reference values, Figure 6 episode facts) are asserted or printed.

Colour meaning, the same in every figure of the paper: blue = success / the controller, vermilion = failure or the expendable rocket, green = the safe region, grey / black = limits and references.
"""
from __future__ import annotations

import csv
import math
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_assets" / "figures"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper_assets" / "montecarlo"))

W_IN = 16 / 2.54                                   # 16 cm
BLUE, VERM, ORANGE, GREEN, SKY, BLACK, GREY, NAVY = "#0072B2", "#D55E00", "#E69F00", "#009E73", "#56B4E9", "#000000", "#7f7f7f", "#1F3A5F"      # Okabe-Ito (colour-blind safe)
DGREEN, DVERM = "#00694b", "#a64800"               # darker text versions of green and vermilion (readable on white)
_names = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((n for n in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans") if n in _names), "sans-serif")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [FONT], "mathtext.fontset": "stixsans", "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.unicode_minus": True, "svg.fonttype": "path"})


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    fig.savefig(OUT / f"{stem}.svg")
    plt.close(fig)


def along_curve_label(ax, xs, ys, x_at, text, side, color, offset_pt=5.0, fs=8, weight="normal"):
    """text that runs along a curve (rotated to its local slope), placed just above (side=+1) or below (side=-1) it; call after the layout is final"""
    i = int(np.argmin(np.abs(np.asarray(xs) - x_at)))
    j0, j1 = max(i - 3, 0), min(i + 3, len(xs) - 1)
    p0, p1 = ax.transData.transform((xs[j0], ys[j0])), ax.transData.transform((xs[j1], ys[j1]))
    ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    if ang > 90:
        ang -= 180
    if ang < -90:
        ang += 180
    th = math.radians(ang)
    dx, dy = -math.sin(th) * offset_pt * side, math.cos(th) * offset_pt * side
    ax.annotate(text, (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", rotation=ang, rotation_mode="anchor", ha="center", va="center", fontsize=fs, color=color, fontweight=weight,
                zorder=7)


# ------------------------------------------------------------------ figure 5
def fig5():
    rows = [r for r in csv.DictReader(open(ROOT / "paper_assets" / "retrain" / "episodes.csv")) if r["controller"] == "sx_retrained"]
    assert len(rows) == 1200
    counts = [sum(int(r["success"]) for r in rows if int(r["level"]) == lv) for lv in range(4)]
    assert counts == [300, 259, 161, 90], f"success counts differ from the paper: {counts}"
    titles = ["L0 – no wind: 300 of 300 landed", "L1 – 5 m/s wind: 259 of 300", "L2 – 10 m/s wind: 161 of 300", "L3 – 15 m/s wind: 90 of 300"]
    fig, axs = plt.subplots(2, 2, figsize=(W_IN, 6.2), sharex=True, sharey=True)
    for lv, ax in enumerate(axs.ravel()):
        rs = [r for r in rows if int(r["level"]) == lv]
        x = np.array([float(r["final_x_m"]) for r in rs])
        v = np.array([float(r["touchdown_speed_mps"]) for r in rs])
        ok = np.array([int(r["success"]) == 1 for r in rs])
        ax.add_patch(plt.Rectangle((-10, 0), 20, 3, fc=GREEN, ec="none", alpha=0.18, zorder=0))              # success box: within 10 m of the pad centre and at most 3 m/s
        ax.axvspan(-10, 10, color=GREY, alpha=0.10, lw=0, zorder=0)
        ax.axhline(3, color=GREEN, lw=1.0, ls="--", zorder=1)
        ax.scatter(x[~ok], v[~ok], s=14, marker="x", c=VERM, lw=0.9, alpha=0.85, zorder=3)
        ax.scatter(x[ok], v[ok], s=14, marker="o", fc=BLUE, ec="white", lw=0.3, alpha=0.95, zorder=4)
        ax.set_xscale("symlog", linthresh=10, linscale=0.55)
        ax.set_yscale("symlog", linthresh=3, linscale=0.8)
        ax.set_xlim(-430, 430)
        ax.set_ylim(0, 45)
        ax.set_xticks([-300, -100, -30, -10, 0, 10, 30, 100, 300])
        ax.set_xticklabels(["−300", "−100", "−30", "−10", "0", "10", "30", "100", "300"], fontsize=8)
        ax.set_yticks([0, 1, 3, 10, 30])
        ax.set_yticklabels(["0", "1", "3", "10", "30"], fontsize=8)
        ax.minorticks_off()
        ax.grid(color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(titles[lv], loc="left", fontsize=9, fontweight="bold", pad=4)
        ax.text(0, 0.62, "SAFE\nLANDING", ha="center", va="center", fontsize=8, fontweight="bold", color=DGREEN, zorder=6, linespacing=1.05)
        ax.text(0.015, 3.5, "Too fast", ha="left", va="bottom", fontsize=8, fontweight="bold", color=DVERM, zorder=6, transform=ax.get_yaxis_transform())
    for xf, ha in ((0.03, "left"), (0.97, "right")):
        axs[1, 1].text(xf, 0.62, "Missed\nthe pad", ha=ha, va="center", fontsize=8, fontweight="bold", color=DVERM, zorder=6, linespacing=1.05, transform=axs[1, 1].get_yaxis_transform())
    axs[0, 0].text(0.03, 0.93, "More wind pushes landings\noff the pad and makes\nthem too fast", ha="left", va="top", fontsize=9, fontweight="bold", color="#222222", linespacing=1.25, zorder=6, transform=axs[0, 0].transAxes)
    for ax in axs[1]:
        ax.set_xlabel("How far from the pad centre (m)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Speed at touchdown (m/s)")
    handles = [Line2D([], [], marker="o", ls="", mfc=BLUE, mec="white", ms=6, label="Success"), Line2D([], [], marker="x", ls="", mec=VERM, ms=6, mew=1.2, label="Failure"),
               Patch(fc=GREEN, alpha=0.3, label="Success region: within 10 m and at most 3 m/s")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.2, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.095, 1, 1), h_pad=0.9, w_pad=0.8)
    fig.text(0.5, 0.062, "Scales are stretched near 0", ha="center", va="center", fontsize=8, color=GREY, style="italic")
    save(fig, "fig5_landing_scatter")
    print("fig5: success counts", counts, "| failure x range", min(float(r["final_x_m"]) for r in rows), max(float(r["final_x_m"]) for r in rows))


# ------------------------------------------------------------------ figure 6
def fig6():
    from stable_baselines3 import PPO
    from src.sim import vehicle as V
    from src.sim.env import SpaceXEnv
    from src.sim import expert as E
    from src.training import reward as R
    from src.training.weights_io import resolve_checkpoint
    from src.utils import config as C
    model = PPO.load(str(resolve_checkpoint("rt1_final")), device="cpu")
    env = SpaceXEnv(level=1)
    env.set_rules_stage(2)
    obs, _ = env.reset(seed=122)
    log = []
    done = False
    while not done:
        s = env.s
        h = max(s.foot_height(), 0.0)
        a_max = V.thrust_full(h) / s.mass - C.GRAVITY_MPS2
        a_plan = max(E.A_PLAN_FRACTION * a_max, 2.0)
        v_des = math.sqrt(2.0 * (a_plan + 0.4 * E._drag_decel(s, h)) * h + E.V_TOUCH ** 2)     # the scripted expert's braking profile evaluated on this lander's state
        log.append((env.t, h, -s.vy, s.vx, s.thr, v_des))
        obs, _, te, tr, info = env.step(model.predict(obs, deterministic=True)[0])
        done = te or tr
    m = info["episode_metrics"]
    log = np.array(log)
    t_td = int(m["touchdown_step"])
    i_td = int(np.argmin(np.abs(log[:, 0] - t_td))) if m["touchdown"] else len(log) - 1
    h_td, vd_td = log[i_td, 1], log[i_td, 2]
    on = np.where(log[:, 4] > 0)[0]
    i_on = int(on[0])
    h_on = float(log[i_on, 1])
    speed_td = float(m["speed_mps"])
    assert bool(m["success"]) and m["end"] == "rest"
    print(f"fig6: level 1 seed 122, rt1_final: success={bool(m['success'])} end={m['end']} first-contact speed {speed_td:.3f} m/s (vertical {vd_td:.3f}), tilt {m['tilt_rad']:.4f} rad, "
          f"pad offset {m['dist_m']:.3f} m, {len(log)} steps; engine first fires at {h_on:.1f} m height (falling speed {log[i_on, 2]:.1f} m/s)")
    brake_max, env_a, env_v0 = C.BRAKE_MAX_MPS2, R.ENVELOPE_DECEL, R.ENVELOPE_FLOOR
    LEG_IN = 0.52                                  # room under the two panels for the legend; the panels keep the size they had at 3.7 in
    fig, axs = plt.subplots(1, 2, figsize=(W_IN, 3.7 + LEG_IN), gridspec_kw=dict(width_ratios=[1.75, 1]))
    keep_traj = np.arange(len(log)) <= i_td
    for k, ax in enumerate(axs):
        hmax = 3300 if k == 0 else 30
        hg = np.linspace(0, hmax, 1500)
        sel = log[:, 1] <= hmax
        # shade the safe speed band
        ax.axhspan(0, C.SUCCESS_MAX_SPEED_MPS, color=GREEN, alpha=0.20, lw=0, zorder=0)
        ax.plot(hg, np.sqrt(2 * brake_max * hg), color=BLACK, lw=1.0, ls="--", zorder=2)
        ax.plot(hg, np.sqrt(2 * env_a * hg) + env_v0, color=GREY, lw=1.4, ls=":", zorder=2)
        ax.plot(log[sel, 1], log[sel, 5], color=SKY, lw=1.3, ls="-.", zorder=3)
        ax.plot(log[keep_traj & sel, 1], log[keep_traj & sel, 2], color=BLUE, lw=2.2, zorder=4)
        ax.plot([h_td], [vd_td], marker="D", color=BLACK, ms=6, ls="none", zorder=6, clip_on=False)
        ax.set_xlim(hmax, 0)
        ax.set_xlabel("Height above the ground (m)")
        ax.grid(color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)
        ax.set_ylabel("Falling speed (m/s)")
    axs[0].set_ylim(0, 360)
    axs[1].set_ylim(0, 34)
    axs[0].plot([h_on], [log[i_on, 2]], marker="^", color=SKY, mec=BLACK, mew=0.7, ms=8, ls="none", zorder=6)
    fig.tight_layout(rect=(0, LEG_IN / (3.7 + LEG_IN), 1, 1), w_pad=1.2)
    # legend under the panels (same line styles and colours as the curves); it replaces the rotated curve labels
    handles = [Line2D([], [], color=BLUE, lw=2.2, ls="-", label="Final controller"),
               Line2D([], [], color=SKY, lw=1.3, ls="-.", label="Expert's braking profile"),
               Line2D([], [], color=BLACK, lw=1.0, ls="--", label="Maximum braking (brake limit)"),
               Line2D([], [], color=GREY, lw=1.4, ls=":", label="Speed envelope"),
               Patch(fc=GREEN, ec="none", alpha=0.20, label="Safe speed (3 m/s or less)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=1.0, fontsize=8, handlelength=2.8, handletextpad=0.6,
               columnspacing=1.6, borderpad=0.5, labelspacing=0.45, bbox_to_anchor=(0.5, 0.012))
    axs[0].annotate(f"Engine fires at\nabout {int(round(h_on, -1))} m", (h_on, log[i_on, 2]), xytext=(1550, 85), textcoords="data", fontsize=8, color="#222222", ha="center", va="center",
                    arrowprops=dict(arrowstyle="-|>", color="#222222", lw=0.8, shrinkA=2, shrinkB=4), zorder=7)
    axs[0].text(3270, 12, "The controller brakes just late\nenough to arrive below 3 m/s", ha="left", va="bottom", fontsize=9, fontweight="bold", color="#222222", linespacing=1.25, zorder=7)
    axs[0].text(0.97, 0.965, "(a) whole descent", transform=axs[0].transAxes, fontsize=8, ha="right", va="top", color="#444444")
    axs[1].text(0.97, 0.965, "(b) last 30 m", transform=axs[1].transAxes, fontsize=8, ha="right", va="top", color="#444444")
    axs[1].text(27, 1.5, "safe speed", ha="left", va="center", fontsize=8, color=DGREEN, style="italic", zorder=7)
    axs[1].annotate(f"Touchdown at {vd_td:.1f} m/s\n({speed_td:.1f} m/s overall;\nlimit 3 m/s)", (h_td, vd_td), xytext=(17.5, 9.0), textcoords="data", fontsize=8, color="#222222",
                    ha="center", va="center", arrowprops=dict(arrowstyle="-|>", color="#222222", lw=0.8, shrinkA=2, shrinkB=4), zorder=7)
    save(fig, "fig6_descent_phase_portrait")
    return m


# ------------------------------------------------------------------ figure 7
def fig7():
    import emissions_mc as M
    d = M.draw()
    X = d["P1"] + d["P2"] + d["F"] + d["T"]
    exp_mean = float(X.mean())
    grid = np.linspace(0.24, 1.00, 305)
    mean, lo, hi = [], [], []
    for p in grid:
        c, _ = M.per_launch(d, float(p))
        mean.append(c.mean())
        lo.append(np.percentile(c, 5))
        hi.append(np.percentile(c, 95))
    mean, lo, hi = map(np.array, (mean, lo, hi))

    def at(p):
        c, _ = M.per_launch(d, p)
        return float(c.mean()), float(np.percentile(c, 5)), float(np.percentile(c, 95))
    print(f"fig7: expendable mean {exp_mean:.2f} t")
    for p in (0.60, 0.90, 0.99, 0.300):
        m_, a_, b_ = at(p)
        print(f"   p={p:.3f}: mean {m_:.1f} t (5th {a_:.1f}, 95th {b_:.1f}), saving {100 * (exp_mean - m_) / exp_mean:.1f} %")
    summ = {int(r["level"]): r for r in csv.DictReader(open(ROOT / "paper_assets" / "retrain" / "summary.csv")) if r["controller"] == "sx_retrained"}
    record = at(0.9963)[0]
    saving_record = 100 * (exp_mean - record) / exp_mean
    print(f"   Falcon 9 record p=0.9963: mean {record:.1f} t, saving {saving_record:.1f} % ({exp_mean - record:.1f} t)")
    assert round(saving_record) == 38 and 998 <= exp_mean < 999, (saving_record, exp_mean)
    fig, ax = plt.subplots(figsize=(W_IN, 4.1))
    ax.fill_between(grid, lo, hi, color=SKY, alpha=0.35, lw=0, zorder=1)
    ax.plot(grid, mean, color=BLUE, lw=1.9, zorder=3)
    ax.axhline(exp_mean, color=VERM, ls="--", lw=1.4, zorder=2)
    ax.set_xlim(0.24, 1.0)
    ax.set_ylim(0, 1100)
    ax.axvline(0.9963, color=GREY, lw=0.9, ls=":", zorder=1)
    # points with their 95 % Wilson intervals
    pts = {}
    for lv, mk in ((1, "o"), (2, "s"), (3, "^")):
        r = summ[lv]
        p, wl, wu = float(r["success_rate"]), float(r["wilson_lower"]), float(r["wilson_upper"])
        y = at(p)[0]
        pts[lv] = (p, y)
        ax.errorbar([p], [y], xerr=[[p - wl], [wu - p]], fmt=mk, color=NAVY, ms=6.5, capsize=3, elinewidth=1.3, zorder=6, mfc=NAVY, mec="white", mew=0.8)
        print(f"   L{lv}: p={p:.3f} interval {wl:.3f}-{wu:.3f}: mean {y:.1f} t, saving {100 * (exp_mean - y) / exp_mean:.1f} %")
    y0 = at(0.99)[0]
    ax.plot([0.99], [y0], marker="D", color=NAVY, mec="white", mew=0.8, ms=6, ls="none", zorder=6)
    ax.annotate("", xy=(0.9985, y0), xytext=(0.99, y0), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2), zorder=6)
    # direct labels
    ax.text(0.26, exp_mean + 18, "Expendable rocket: 998 t", color=DVERM, fontsize=9, fontweight="bold", ha="left", va="bottom")
    ax.text(0.665, 548, "Average of 200,000 simulations", color=BLUE, fontsize=8, fontweight="bold", ha="center", va="center", zorder=7)
    ax.annotate("", xy=(0.64, at(0.64)[0] - 6), xytext=(0.64, 575), arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8, shrinkA=0, shrinkB=0), zorder=6)
    ax.text(0.385, 648, "90 % of simulations fall in this band", color="#2b7fb0", fontsize=8, ha="center", va="center", zorder=7)
    ax.annotate("", xy=(0.42, lo[np.argmin(abs(grid - 0.42))] - 2), xytext=(0.42, 672), arrowprops=dict(arrowstyle="-", color="#2b7fb0", lw=0.8, shrinkA=0, shrinkB=0), zorder=6)
    ax.text(0.255, pts[3][1] - 105, "L3, 15 m/s wind: 0.30", fontsize=8, ha="left", va="center", color="#222222", zorder=7)
    ax.text(pts[2][0], pts[2][1] + 98, "L2, 10 m/s: 0.54", fontsize=8, ha="center", va="center", color="#222222", zorder=7)
    ax.text(pts[1][0] - 0.065, pts[1][1] + 112, "L1, 5 m/s wind: 0.86", fontsize=8, ha="center", va="center", color="#222222", zorder=7)
    ax.text(0.985, y0 - 95, "L0: 300 of 300 landed,\nso p is at least 0.99", fontsize=8, ha="right", va="center", color="#222222", zorder=7, linespacing=1.15)
    ax.text(0.9935, 1040, "Falcon 9 record (0.9963)", fontsize=8, ha="right", va="center", color="#444444", zorder=7)
    ax.text(0.255, 25, "bars = 95 % confidence interval", fontsize=8, color=GREY, ha="left", va="bottom")
    # the reduction at the real record
    xa = 0.978
    ax.annotate("", xy=(xa, record), xytext=(xa, exp_mean), arrowprops=dict(arrowstyle="<|-|>", color="#222222", lw=1.2, shrinkA=0, shrinkB=0, mutation_scale=9), zorder=7)
    ax.text(xa - 0.008, (record + exp_mean) / 2, f"{saving_record:.0f} % less\nCO$_2$e", ha="right", va="center", fontsize=9, fontweight="bold", color="#222222", linespacing=1.15,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9), zorder=8)
    ax.text(0.26, 235, "The better the booster lands,\nthe less CO$_2$e each launch emits", fontsize=10, fontweight="bold", color="#222222", ha="left", va="center", linespacing=1.25, zorder=7)
    ax.set_xlabel("Landing success rate (p)")
    ax.set_ylabel("CO$_2$e per launch (tonnes)")
    ax.set_yticks(range(0, 1101, 200))
    ax.grid(color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig7_emissions_vs_p")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig5()
    fig6()
    fig7()
    from PIL import Image
    for f in ("fig5_landing_scatter.png", "fig6_descent_phase_portrait.png", "fig7_emissions_vs_p.png"):
        im = Image.open(OUT / f)
        print(f, im.size, f"{im.size[0] / 300 * 2.54:.1f} x {im.size[1] / 300 * 2.54:.1f} cm", "| font:", FONT)
