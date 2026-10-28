"""Three figures for the GENIUS Olympiad proposal REBOUND (matplotlib only).

    python paper_assets/proposal/make_proposal_figs.py --lang en
    python paper_assets/proposal/make_proposal_figs.py --lang id

Writes into paper_assets/proposal/ (PNG at 300 dpi and SVG for each):
    fig2_1_expected_flights_<lang>   expected flights per booster against landing success probability (5.5 x 2.4 in)
    fig3_3a_chain_<lang>             flow diagram: success probability -> flights -> emissions -> comparison (2.6 x 2.9 in)
    fig3_3b_training_<lang>          flow diagram: expert -> behaviour cloning -> DAgger loop -> trained controller (2.6 x 2.9 in)
    fig2_1_values.csv                p, E(N=15), E(N=25), E(N=40) on a 0.005 grid
The figures have exactly the stated size in inches (the layout is fitted to the canvas instead of cropping with a tight bounding box, so the printed size stays as asked).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN, DARKRED, GREY = "#1f4e8c", "#b5651d", "#2e7d32", "#7b1e1e", "#8c8c8c"
DPI = 300

_names = {f.name for f in font_manager.fontManager.ttflist}
SERIF = next((n for n in ("Times New Roman", "Liberation Serif", "DejaVu Serif") if n in _names), "serif")
plt.rcParams.update({"font.family": "serif", "font.serif": [SERIF], "mathtext.fontset": "stix", "svg.fonttype": "path", "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "axes.unicode_minus": True})

TXT = {
    "en": dict(
        xlabel="Landing success probability p", ylabel="Expected flights per booster E", falcon="Falcon fleet record", sim="Simulator",
        chain=[("Landing success probability p", "measured in the simulator, 300 episodes per level"),
               ("Expected flights per booster E", r"E = (1 $-$ p$^{N}$) / (1 $-$ p)"),
               ("Emissions per launch C", r"C = P$_1$(1 + r(E $-$ 1))/E + P$_2$ + F + T"),
               ("Comparison and robustness", "vs expendable launch; sensitivity, black carbon and rebound break-even")],
        train=[("Scripted expert controller", "tuned landing rules"), ("Behavior cloning", "network copies the expert"),
               ("DAgger loop", "network flies, expert relabels the states it reaches, retrain"),
               ("Trained controller", "tested on seeds 0 to 299 at levels L0 to L3")],
        repeat="repeat"),
    "id": dict(
        xlabel="Probabilitas keberhasilan pendaratan p", ylabel="Jumlah penerbangan yang diharapkan\nper booster E", falcon="Rekor armada Falcon", sim="Simulator",
        chain=[("Probabilitas keberhasilan pendaratan p", "diukur di simulator, 300 episode per tingkat"),
               ("Jumlah penerbangan yang diharapkan E", r"E = (1 $-$ p$^{N}$) / (1 $-$ p)"),
               ("Emisi per peluncuran C", r"C = P$_1$(1 + r(E $-$ 1))/E + P$_2$ + F + T"),
               ("Perbandingan dan ketahanan", "dengan peluncuran sekali pakai; sensitivitas, titik impas karbon hitam dan efek rebound")],
        train=[("Pengendali ahli berskrip", "aturan pendaratan yang disetel"), ("Kloning perilaku", "jaringan menyalin pengendali ahli"),
               ("Putaran DAgger", "jaringan terbang, pengendali ahli melabeli ulang keadaan yang dicapai, latih ulang"),
               ("Pengendali terlatih", "diuji pada benih 0 sampai 299 di tingkat L0 sampai L3")],
        repeat="ulangi"),
}


def expected_flights(p, N):
    """E = (1 - p^N) / (1 - p); the limit at p = 1 is N."""
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    one = p >= 1.0
    out[one] = N
    out[~one] = (1.0 - p[~one] ** N) / (1.0 - p[~one])
    return out


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=DPI)
    fig.savefig(OUT / f"{stem}.svg")
    plt.close(fig)


# ------------------------------------------------------------------ figure 2.1
def fig_flights(lang):
    t = TXT[lang]
    assert abs(float(expected_flights([0.5], 40)[0]) - 2.0) < 1e-6
    assert abs(float(expected_flights([0.9], 15)[0]) - 7.94) < 0.005
    assert float(expected_flights([1.0], 25)[0]) == 25
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 3)
    with open(OUT / "fig2_1_values.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "E(N=15)", "E(N=25)", "E(N=40)"])
        for p, a, b, c in zip(grid, expected_flights(grid, 15), expected_flights(grid, 25), expected_flights(grid, 40)):
            w.writerow([repr(float(p)), repr(float(a)), repr(float(b)), repr(float(c))])
    fine = np.linspace(0.0, 1.0, 1001)
    fig = plt.figure(figsize=(5.5, 2.4), dpi=DPI)
    ax = fig.add_axes([0.105 if lang == "en" else 0.135, 0.20, 0.80 if lang == "en" else 0.77, 0.77])
    for N, col in ((15, BLUE), (25, ORANGE), (40, GREEN)):
        ax.plot(fine, expected_flights(fine, N), color=col, lw=1.6)
        ax.annotate(f"N = {N}", (1.0, N), xytext=(3, 0), textcoords="offset points", color=col, fontsize=9, va="center", ha="left", annotation_clip=False)
    marks = [(0.300, "L3", 33, "center"), (0.537, "L2", 33, "center"), (0.863, "L1", 26, "right"), (0.9963, None, 38.5, "right")]
    for p, lv, y, ha in marks:
        ax.plot([p, p], [0, y if lv else 42], color=GREY, lw=0.8, ls=":", zorder=1)      # the dotted line stops under its own label
        label = f"{t['sim']} {lv}" if lv else t["falcon"]
        ax.text(p + (-0.008 if ha == "right" else 0), y, label, fontsize=9, color="black" if lv else DARKRED, ha=ha, va="center",
                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.9), zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 42)
    ax.set_xlabel(t["xlabel"], fontsize=9.5, labelpad=2)
    ax.set_ylabel(t["ylabel"], fontsize=9.5 if lang == "en" else 9, labelpad=2, linespacing=1.05)
    ax.tick_params(labelsize=9, length=3, pad=2)
    ax.grid(color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    save(fig, f"fig2_1_expected_flights_{lang}")



# ------------------------------------------------------------------ figure 1 (English, plain-language version, same style as figures 4 to 7)
def fig_flights_en():
    """Figure 1 of the paper: sans-serif, 16 x 7 cm, percent axis, direct labels, values read from the formula (no number is typed in)."""
    OB = dict(blue="#0072B2", sky="#56B4E9", navy="#1F3A5F", vermilion="#D55E00", green="#009E73", grey="#7f7f7f")
    names = {f.name for f in font_manager.fontManager.ttflist}
    sans = next((n for n in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans") if n in names), "sans-serif")
    assert abs(float(expected_flights([0.5], 40)[0]) - 2.0) < 1e-6
    assert abs(float(expected_flights([0.9], 15)[0]) - 7.94) < 0.005
    assert float(expected_flights([1.0], 25)[0]) == 25
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 3)
    with open(OUT / "fig2_1_values.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "E(N=15)", "E(N=25)", "E(N=40)"])
        for p_, a, b, c in zip(grid, expected_flights(grid, 15), expected_flights(grid, 25), expected_flights(grid, 40)):
            w.writerow([repr(float(p_)), repr(float(a)), repr(float(b)), repr(float(c))])
    pts = [(0.30, "L3, 15" + " m/s wind", 4.6), (0.54, "L2, 10" + " m/s", 8.6), (0.86, "L1, 5" + " m/s", None), (0.99, "99 %", None), (0.9963, "Falcon 9 record", None)]
    vals = {p_: float(expected_flights([p_], 25)[0]) for p_, _, _ in pts}
    print("fig1: expected flights for N = 25: " + ", ".join(f"p={p_:g}: {v:.2f}" for p_, v in vals.items()))
    print("fig1: flight limit hardly matters below 80 %: E(0.8) for N = 15, 25, 40 =", [round(float(expected_flights([0.8], n)[0]), 2) for n in (15, 25, 40)])
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": [sans], "mathtext.fontset": "stixsans", "svg.fonttype": "path", "axes.unicode_minus": True}):
        fig = plt.figure(figsize=(16 / 2.54, 7 / 2.54), dpi=DPI)
        ax = fig.add_axes([0.085, 0.175, 0.725, 0.80])
        fine = np.linspace(0.0, 1.0, 1001)
        for n, col, lw in ((15, OB["sky"], 1.6), (25, OB["blue"], 2.2), (40, OB["navy"], 1.6)):
            ax.plot(fine * 100, expected_flights(fine, n), color=col, lw=lw, zorder=3)
            ax.annotate(f"limit {n} flights", (100, n), xytext=(4, 0), textcoords="offset points", color=col if n != 15 else "#2b7fb0", fontsize=8, va="center", ha="left", annotation_clip=False,
                        fontweight="bold" if n == 25 else "normal")
        # the shaded stretch where the limit does not matter
        ax.annotate("", xy=(80, 13.5), xytext=(1, 13.5), arrowprops=dict(arrowstyle="<|-|>", color=OB["grey"], lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=6), zorder=2)
        ax.text(40.5, 15.0, "Below about 80 % the flight limit hardly matters", fontsize=8, color="#555555", ha="center", va="bottom", style="italic")
        # dots on the N = 25 curve with their values
        for p_, lab, ytxt in pts:
            ax.plot([p_ * 100], [vals[p_]], marker="o", ms=5, color=OB["blue"], mec="white", mew=0.8, zorder=6, clip_on=False)
        ax.annotate("L3, 15 m/s wind: " + f"{vals[0.30]:.1f}", (30, vals[0.30]), xytext=(30, 4.8), fontsize=8, ha="center", va="center", color="#222222",
                    arrowprops=dict(arrowstyle="-", color=OB["grey"], lw=0.6, shrinkA=0, shrinkB=2))
        ax.annotate("L2, 10 m/s: " + f"{vals[0.54]:.1f}", (54, vals[0.54]), xytext=(54, 8.4), fontsize=8, ha="center", va="center", color="#222222",
                    arrowprops=dict(arrowstyle="-", color=OB["grey"], lw=0.6, shrinkA=0, shrinkB=2))
        ax.annotate("L1, 5 m/s: " + f"{vals[0.86]:.1f}" + " flights", (86, vals[0.86]), xytext=(75.5, 10.4), fontsize=8, ha="center", va="center", color="#222222",
                    arrowprops=dict(arrowstyle="-", color=OB["grey"], lw=0.6, shrinkA=0, shrinkB=2))
        ax.annotate("99 %: " + f"{vals[0.99]:.1f}" + " flights", (99, vals[0.99]), xytext=(84, 18.5), fontsize=8, ha="center", va="center", color="#222222",
                    arrowprops=dict(arrowstyle="-", color=OB["grey"], lw=0.6, shrinkA=0, shrinkB=2))
        ax.annotate("Falcon 9 record (99.63 %): " + f"{vals[0.9963]:.1f}", (99.63, vals[0.9963]), xytext=(78, 27.0), fontsize=8, ha="center", va="center", color=OB["navy"],
                    fontweight="bold", arrowprops=dict(arrowstyle="-", color=OB["navy"], lw=0.7, shrinkA=0, shrinkB=2))
        ax.text(1.5, 37.5, "A booster only flies many times\nif it almost always lands", fontsize=9, fontweight="bold", color="#222222", ha="left", va="top")
        ax.text(1.5, 30.0, r"$E=\sum_{k=0}^{N-1}p^{k}$", fontsize=9, color="#222222", ha="left", va="top")
        ax.text(1.5, 22.0, "p = landing success rate\nN = flight limit", fontsize=8, color="#555555", ha="left", va="top")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 42)
        ax.set_xticks(range(0, 101, 20))
        ax.set_xlabel("Landing success rate p (%)", fontsize=9, labelpad=2)
        ax.set_ylabel("Expected flights per booster (E)", fontsize=9, labelpad=2)
        ax.tick_params(labelsize=8, length=3, pad=2)
        ax.grid(color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)
        save(fig, "fig2_1_expected_flights_en")


# ------------------------------------------------------------------ flow diagrams
def text_w(fig, s, fs, weight="normal"):
    r = fig.canvas.get_renderer()
    tx = fig.text(0, 0, s, fontsize=fs, fontweight=weight)
    w = tx.get_window_extent(r).width / fig.dpi
    tx.remove()
    return w


def wrap(fig, s, fs, max_w, weight="normal"):
    lines, cur = [], ""
    for word in s.split(" "):
        trial = (cur + " " + word).strip()
        if cur and text_w(fig, trial, fs, weight) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    lines.append(cur)
    return lines


def flow(lang, items, colors, stem, box_x0, box_w, loop_on=None):
    """items = [(title, body)] drawn top to bottom as rounded boxes joined by arrows; the axes are 1 data unit = 1 inch."""
    W, H = 2.6, 2.9
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    FT, FB, pad = 9.5, 9.0, 0.055
    lh_t, lh_b = FT * 1.2 / 72, FB * 1.2 / 72
    inner = box_w - 2 * 0.06
    boxes = []
    for title, body in items:
        tl = wrap(fig, title, FT, inner, "bold")
        bl = wrap(fig, body, FB, inner)
        h = 2 * pad + len(tl) * lh_t + 0.02 + len(bl) * lh_b
        boxes.append((tl, bl, h))
    margin = 0.06
    gap = (H - 2 * margin - sum(b[2] for b in boxes)) / (len(boxes) - 1)
    assert gap >= 0.09, f"{stem}: boxes do not fit (gap {gap:.3f} in)"
    gap = min(gap, 0.30)
    total = sum(b[2] for b in boxes) + gap * (len(boxes) - 1)
    y = H - (H - total) / 2
    xc = box_x0 + box_w / 2
    rects = []
    for (tl, bl, h), col in zip(boxes, colors):
        ax.add_patch(FancyBboxPatch((box_x0, y - h), box_w, h, boxstyle="round,pad=0,rounding_size=0.07", fc="white", ec=col, lw=1.4))
        ty = y - pad
        for line in tl:
            ax.text(xc, ty, line, fontsize=FT, fontweight="bold", color=col, ha="center", va="top")
            ty -= lh_t
        ty -= 0.02
        for line in bl:
            ax.text(xc, ty, line, fontsize=FB, color="black", ha="center", va="top")
            ty -= lh_b
        rects.append((y, y - h))
        y -= h + gap
    for (_, b0), (t1, _) in zip(rects[:-1], rects[1:]):
        ax.add_patch(FancyArrowPatch((xc, b0 - 0.005), (xc, t1 + 0.005), arrowstyle="-|>", mutation_scale=11, lw=1.3, color=GREY, shrinkA=0, shrinkB=0))
    if loop_on is not None:
        top, bot = rects[loop_on]
        yc, d = (top + bot) / 2, min(0.22, (top - bot) / 2 - 0.12)
        xr = box_x0 + box_w
        ax.add_patch(FancyArrowPatch((xr + 0.005, yc - d), (xr + 0.005, yc + d), connectionstyle="arc3,rad=1.25", arrowstyle="-|>", mutation_scale=11, lw=1.5, color=ORANGE,
                                     shrinkA=0, shrinkB=0))
        ax.text(xr + 0.27 + 0.1, yc, TXT[lang]["repeat"], fontsize=9, color=ORANGE, rotation=90, ha="center", va="center")
    save(fig, stem)
    return gap


def fig_chain(lang):
    flow(lang, TXT[lang]["chain"], [BLUE, BLUE, ORANGE, GREEN], f"fig3_3a_chain_{lang}", 0.09, 2.42)


def fig_training(lang):
    flow(lang, TXT[lang]["train"], [BLUE, BLUE, ORANGE, GREEN], f"fig3_3b_training_{lang}", 0.10, 1.90, loop_on=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "id"], required=True)
    a = ap.parse_args()
    (fig_flights_en if a.lang == "en" else fig_flights)(a.lang) if a.lang != "en" else fig_flights_en()
    fig_chain(a.lang)
    fig_training(a.lang)
    from PIL import Image
    for p in sorted(OUT.glob(f"*_{a.lang}.png")):
        print(p.name, Image.open(p).size, f"font: {SERIF}")


if __name__ == "__main__":
    main()
