"""Figure 4 of the REBOUND paper: the method as a flow diagram, with symbols, a plain-English line under every box title and numbered steps (Python + Matplotlib only).

    python paper_assets/figures/make_fig4_method.py

Writes fig4_method.png (300 dpi), fig4_method.svg and fig4_method.pdf to paper_assets/figures/. The canvas is exactly 16 cm wide, so nothing is rescaled in the document.
Panel (a): the analysis chain (input, data processing, AI model, analysis, output), steps 1 to 8, with the symbols of Section 3.2 and a grey legend; panel (b): how the controller is trained, steps 1 to 4.
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
W_IN = 16 / 2.54
H_CM = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0                       # the canvas height in cm (10 cm unless the layout needs more; see the printed use)
H_IN = H_CM / 2.54
TEXT, LINE, FILL, EDGE, NAVY, STAGE, PLAIN, PLAIN_ON_NAVY, STEP = "#222222", "#555555", "#F4F5F7", "#444444", "#1F3A5F", "#707070", "#666666", "#C9D3E0", "#8C8C8C"
_names = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((n for n in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans") if n in _names), "sans-serif")
# math in STIX Sans, a sans-serif face that sits well next to the Arial body text
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [FONT], "mathtext.fontset": "stixsans", "svg.fonttype": "path", "pdf.fonttype": 42,
                     "figure.facecolor": "white", "savefig.facecolor": "white", "axes.unicode_minus": True})
FT, FD, FS, FL = 8.5, 8.0, 8.0, 10.0           # box title, box text, stage label and legend, panel letter (pt)
TS, EN = " ", "–"                      # thin space (before units and percent signs), en dash for ranges

fig = plt.figure(figsize=(W_IN, H_IN), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_IN)
ax.set_ylim(0, H_IN)
ax.axis("off")
R = fig.canvas.get_renderer()
inch = lambda artist: (lambda b: (b.x0 / 300, b.y0 / 300, b.x1 / 300, b.y1 / 300))(artist.get_window_extent(R))
CHECK = []


def text_w(s, fs, weight="normal"):
    t = ax.text(0, 0, s, fontsize=fs, fontweight=weight)
    x0, _, x1, _ = inch(t)
    t.remove()
    return x1 - x0


def words(s):
    """split on spaces that are not inside $...$ (a formula is never broken)"""
    out, cur, inside = [], "", False
    for ch in s:
        if ch == "$":
            inside = not inside
        if ch == " " and not inside:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [w for w in out if w]


def wrap(s, fs, max_w, weight="normal"):
    lines, cur = [], ""
    for word in words(s):
        trial = (cur + " " + word).strip()
        if cur and text_w(trial, fs, weight) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    lines.append(cur)
    return lines


X0, GAP = 0.30, 0.17
COL_W = (W_IN - X0 - 0.06 - 3 * GAP) / 4
col_x = [X0 + k * (COL_W + GAP) for k in range(4)]
PAD, LS, INNER_M = 0.04, 1.14, 0.04


def content(title, plain, body, w):
    inner = w - 2 * INNER_M
    tl = wrap(title, FT, inner, "bold")
    pl = wrap(plain, FD, inner) if plain else []
    bl = [x for part in body.split("\n") for x in wrap(part, FD, inner)] if body else []
    return tl, pl, bl


def measure(lines, fs):
    hs = []
    for ln in lines:
        t = ax.text(0, 0, ln, fontsize=fs)
        _, y0, _, y1 = inch(t)
        t.remove()
        extra = 0.05 if "dfrac" in ln else 0.035 if "sum" in ln else 0.0         # fractions and sums need room for their limits
        hs.append(max(y1 - y0, fs * LS / 72) + extra)
    return hs


def block_height(tl, pl, bl):
    return sum(measure(tl, FT)) + (0.015 + sum(measure(pl, FD)) if pl else 0) + (0.025 + sum(measure(bl, FD)) if bl else 0)


def box_height(items):
    return max(2 * PAD + block_height(*content(t, p, b, COL_W)) for t, p, b in items)


def draw_box(x, y_top, h, title, plain, body, step, accent=False):
    yb = H_IN - y_top - h
    ax.add_patch(FancyBboxPatch((x, yb), COL_W, h, boxstyle="round,pad=0,rounding_size=0.006", fc=NAVY if accent else FILL, ec=NAVY if accent else EDGE, lw=0.8, zorder=3))
    tl, pl, bl = content(title, plain, body, COL_W)
    total = block_height(tl, pl, bl)
    y = H_IN - y_top - (h - total) / 2
    cx = x + COL_W / 2
    tc, pc, bc = ("white", PLAIN_ON_NAVY, "#E4E9F0") if accent else (TEXT, PLAIN, TEXT)
    for ln, hh in zip(tl, measure(tl, FT)):
        t = ax.text(cx, y, ln, fontsize=FT, fontweight="bold", color=tc, ha="center", va="top", zorder=5)
        CHECK.append((t, (x, yb, x + COL_W, yb + h)))
        y -= hh
    if pl:
        y -= 0.015
        for ln, hh in zip(pl, measure(pl, FD)):
            t = ax.text(cx, y, ln, fontsize=FD, color=pc, ha="center", va="top", zorder=5)
            CHECK.append((t, (x, yb, x + COL_W, yb + h)))
            y -= hh
    if bl:
        y -= 0.025
        for ln, hh in zip(bl, measure(bl, FD)):
            t = ax.text(cx, y, ln, fontsize=FD, color=bc, ha="center", va="top", zorder=5)
            CHECK.append((t, (x, yb, x + COL_W, yb + h)))
            y -= hh
    # step number: a small grey circle on the top-left corner of the box
    ax.add_patch(Circle((x, H_IN - y_top), 0.085, fc=STEP, ec="white", lw=0.8, zorder=6))
    ax.text(x, H_IN - y_top - 0.004, str(step), fontsize=8, fontweight="bold", color="white", ha="center", va="center", zorder=7)


def arrow(p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=1.0, color=LINE, shrinkA=0, shrinkB=0, zorder=2))


def seg(p0, p1):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=LINE, lw=1.0, solid_capstyle="butt", zorder=2)


# ---- content (symbols as in Section 3.2: p, N, E, P1, P2, F, T, r); each box = (title, plain-English line, symbols / details)
ROW1 = [("Simulator", "a virtual Falcon 9 landing in four wind levels", r"$\bar{w}=0,\,5,\,10,\,15\,\mathrm{m\,s^{-1}}$"),
        ("Scaling and noise", "puts all readings on a similar scale", "noise " + r"$\sigma=0.05\,\%$" + EN + r"$5\,\%$" + " of range"),
        ("Neural network", "the AI that flies the landing", "2" + TS + "×" + TS + "256 tanh, 18 inputs, 5 outputs"),
        ("Landing success", "how often it lands safely", r"$\hat{p}=k/n,\ n=300$" + "\nper level, 95" + TS + "% Wilson interval")]
ROW2 = [("Expected flights", "how many times a booster flies", r"$E=\sum_{k=0}^{N-1}p^{k}$" + "\n" + r"reuse limit $N\in[15,\,40]$"),
        ("Emissions per launch", "CO$_2$e of one launch", r"$C=\dfrac{P_1\,[\,1+r\,(E-1)\,]}{E}$" + "\n" + r"$+\,P_2+F+T$"),
        ("Sensitivity", "which inputs matter most", r"Spearman $\rho_{s}$ between each input and $C$"),
        ("Break-even", "what cancels the saving", r"launch growth $1/\kappa\approx1.6\times$" + "\n" + r"$\kappa=C_{\mathrm{reuse}}/C_{\mathrm{exp}}$" + "\n" + r"black carbon $m_{\mathrm{BC}}\approx0.24\,\mathrm{kg}$")]
PB = [("Scripted expert", "hand-written landing rules", ""), ("Behaviour cloning", "the network copies the expert", ""),
      ("DAgger loop", "the network flies, the expert corrects it", ""), ("Trained controller", "tested on 300 landings per wind level", "")]
LEGEND = (r"$p$ landing success probability; $N$ reuse limit; $E$ expected flights (closed form $(1-p^{N})/(1-p)$); $P_1$, $P_2$ production emissions of stage 1 and 2; "
          r"$F$ propellant; $T$ transport; $r$ refurbishment share; $k$ successes in $n$ episodes. "
          r"Monte Carlo: $2\times10^{5}$ draws, inputs ~ triangular; $m_{\mathrm{BC}}$ at the warming factor $1.6\times10^{6}$ kg CO$_2$e per kg.")
H_A = box_height(ROW1 + ROW2)
H_B = box_height(PB)
leg_lines = wrap(LEGEND, FS, W_IN - 0.03 - 0.06)
leg_h = [max(h, FS * 1.3 / 72) for h in measure(leg_lines, FS)]
LEG_H = sum(leg_h) + 0.02
GAP_ROW, LOOP_H = 0.29, 0.25
top = 0.04
y_row1 = top + 0.20
y_row2 = y_row1 + H_A + GAP_ROW
y_leg = y_row2 + H_A + 0.06
y_b_letter = y_leg + LEG_H + 0.05
y_b_box = y_b_letter + 0.15
used = y_b_box + H_B + LOOP_H + 0.03
print(f"canvas 16 x {H_CM:g} cm; box heights: panel (a) {H_A:.3f} in, panel (b) {H_B:.3f} in, legend {len(leg_lines)} lines; vertical use {used:.3f} of {H_IN:.3f} in; box width {COL_W:.3f} in")
assert used <= H_IN + 1e-6, f"layout does not fit {H_CM:g} cm (over by {used - H_IN:.3f} in)"
slack = H_IN - used
y_b_letter += slack * 0.5
y_b_box += slack * 0.5

ax.text(0.03, H_IN - top, "(a)", fontsize=FL, fontweight="bold", color=TEXT, ha="left", va="top")
ax.text(0.03, H_IN - y_b_letter, "(b)", fontsize=FL, fontweight="bold", color=TEXT, ha="left", va="top")
# ---- panel (a), row 1: steps 1 to 4
labels1 = ["INPUT", "DATA PROCESSING", "AI MODEL", "ANALYSIS"]
for k, (title, plain, body) in enumerate(ROW1):
    draw_box(col_x[k], y_row1, H_A, title, plain, body, k + 1)
    ax.text(col_x[k] + 0.14, H_IN - y_row1 + 0.03, labels1[k], fontsize=FS, color=STAGE, ha="left", va="bottom")
    if k:
        arrow((col_x[k - 1] + COL_W, H_IN - y_row1 - H_A / 2), (col_x[k], H_IN - y_row1 - H_A / 2))
# ---- panel (a), row 2: steps 5 to 8
labels2 = ["", "", "OUTPUT", ""]
for k, (title, plain, body) in enumerate(ROW2):
    draw_box(col_x[k], y_row2, H_A, title, plain, body, k + 5, accent=(k >= 2))
    if labels2[k]:
        ax.text(col_x[k] + 0.14, H_IN - y_row2 + 0.03, labels2[k], fontsize=FS, color=STAGE, ha="left", va="bottom")
    if k:
        arrow((col_x[k - 1] + COL_W, H_IN - y_row2 - H_A / 2), (col_x[k], H_IN - y_row2 - H_A / 2))
x_out, x_in = col_x[3] + COL_W / 2, col_x[0] + COL_W / 2
y_mid = H_IN - (y_row1 + H_A + 0.10)
seg((x_out, H_IN - y_row1 - H_A), (x_out, y_mid))
seg((x_out, y_mid), (x_in, y_mid))
arrow((x_in, y_mid), (x_in, H_IN - y_row2))
# ---- legend (grey)
yy = H_IN - y_leg
for ln, hh in zip(leg_lines, leg_h):
    t = ax.text(0.03, yy, ln, fontsize=FS, color=STAGE, ha="left", va="top")
    CHECK.append((t, (0.0, 0.0, W_IN, H_IN)))
    yy -= hh
# ---- panel (b): steps 1 to 4
for k, (title, plain, body) in enumerate(PB):
    draw_box(col_x[k], y_b_box, H_B, title, plain, body, k + 1, accent=(k == 3))
    if k:
        arrow((col_x[k - 1] + COL_W, H_IN - y_b_box - H_B / 2), (col_x[k], H_IN - y_b_box - H_B / 2))
cx3 = col_x[2] + COL_W / 2
yb_box = H_IN - y_b_box - H_B
y_u = yb_box - 0.13
seg((cx3 + 0.38, yb_box), (cx3 + 0.38, y_u))
seg((cx3 + 0.38, y_u), (cx3 - 0.38, y_u))
arrow((cx3 - 0.38, y_u), (cx3 - 0.38, yb_box))
t = ax.text(cx3, y_u - 0.02, r"rounds $i=1,\dots,8$", fontsize=FS, color=TEXT, ha="center", va="top")
CHECK.append((t, (0.0, 0.0, W_IN, H_IN)))

worst = 1e9
for t, (x0, y0, x1, y1) in CHECK:
    a, b, c, d = inch(t)
    margin = min(a - x0, x1 - c, b - y0, y1 - d)
    worst = min(worst, margin)
    assert margin >= 0.02, f"text '{t.get_text()[:40]}' is {margin:.3f} in from the box or canvas edge"
print(f"smallest margin between text and box or canvas edge: {worst:.3f} in ({worst * 300:.0f} px at 300 dpi)")
fig.savefig(OUT / "fig4_method.png", dpi=300)
fig.savefig(OUT / "fig4_method.svg")
fig.savefig(OUT / "fig4_method.pdf")
from PIL import Image

im = Image.open(OUT / "fig4_method.png")
print("fig4_method.png", im.size, f"{im.size[0] / 300 * 2.54:.2f} x {im.size[1] / 300 * 2.54:.2f} cm")
import matplotlib as mpl

print("versions:", "Python", platform.python_version(), "| Matplotlib", mpl.__version__, "| NumPy", np.__version__, "| Pillow (only to read the size)", __import__("PIL").__version__, "| font:", FONT,
      font_manager.findfont(FONT))
