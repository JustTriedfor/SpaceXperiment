"""fig4_3_emissions_vs_p.png and results.md from results.csv / emissions_mc.py.   python paper_assets/montecarlo/make_outputs.py"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import emissions_mc as M  # noqa: E402

names = {f.name for f in font_manager.fontManager.ttflist}
SERIF = next((n for n in ("Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "STIXGeneral", "DejaVu Serif") if n in names), "serif")
plt.rcParams.update({"font.family": "serif", "font.serif": [SERIF], "mathtext.fontset": "stix", "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "font.size": 11})
rows = list(csv.DictReader(open(OUT / "results.csv")))
f = lambda r, k: float(r[k])
get = lambda step, start: next(r for r in rows if r["step"] == step and r["label"].startswith(start))

# ---- figure: the same draws as the table, a fine grid of p, no smoothing
d = M.draw()
X = d["P1"] + d["P2"] + d["F"] + d["T"]
grid = np.linspace(0.0, 1.0, 401)
mean, lo, hi = [], [], []
for p in grid:
    c, _ = M.per_launch(d, p)
    mean.append(c.mean())
    lo.append(np.percentile(c, 5))
    hi.append(np.percentile(c, 95))
exp_mean = float(X.mean())
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
ax.fill_between(grid, lo, hi, color="#9db8d9", alpha=0.55, lw=0, label="5th to 95th percentile (Monte Carlo)")
ax.plot(grid, mean, color="#1f4e8c", lw=1.8, label="Mean, reusable booster")
ax.axhline(exp_mean, color="#c00000", ls="--", lw=1.3, label=f"Expendable rocket, mean {exp_mean:.1f} t")
for lv, col, mk in ((1, "#2e7d32", "o"), (2, "#b5651d", "s"), (3, "#7a1f1f", "^")):
    r = get("step2", f"L{lv} measured")
    lw_, up = get("step3", f"L{lv} Wilson lower"), get("step3", f"L{lv} Wilson upper")
    p = f(r, "p")
    ax.errorbar([p], [f(r, "mean_tco2e")], xerr=[[p - f(lw_, "p")], [f(up, "p") - p]], fmt=mk, color=col, ms=6.5, capsize=3, elinewidth=1.2, zorder=5, label=f"Simulator L{lv}: p = {p:.3f} (95 % Wilson interval)")
r = get("step1", "Table 4.1, p = 0.9963")
ax.plot([0.9963], [f(r, "mean_tco2e")], marker="D", color="black", ms=6.5, ls="none", zorder=6, clip_on=False, label="Falcon fleet record, p = 0.9963")
ax.set_xlim(0, 1.0)
ax.set_xlabel("Landing success probability p (dimensionless)")
ax.set_ylabel("Emissions per launch (tonnes CO$_2$e)")
ax.set_title("Emissions per launch against landing success probability (200,000 draws per point)", fontsize=11)
ax.grid(color="#e4e4e4", lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8.3, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "fig4_3_emissions_vs_p.png")
print("figure written", exp_mean)

# ---- results.md
T = lambda r: f"{f(r, 'expected_flights'):.2f} | {f(r, 'mean_tco2e'):.1f} | {f(r, 'p5_tco2e'):.1f} | {f(r, 'p95_tco2e'):.1f} | {f(r, 'saving_percent'):.2f} | {f(r, 'saving_tco2e'):.1f}"
paper = {"0.33": (1.5, 868, 13), "0.6": (2.5, 762, 24), "0.9": (9.3, 647, 35), "0.95": (14.8, 631, 37), "0.99": (24.0, 622, 38), "0.9963": (26.1, 620, 38)}
md = ["# Emissions Monte Carlo results (rebuilt model, 200,000 draws, one seed for every p; full precision in results.csv)", "",
      "Columns: expected flights = mean over the draws (reuse limit N uniform 15 to 40); mean and 5th / 95th percentile of tonnes CO2e per launch; saving against the expendable mean "
      f"({exp_mean:.1f} t on the same draws). Rounding here: flights 2 decimals, tonnes 1 decimal, percent 2 decimals.", "",
      "## Step 1: reproduction of Table 4.1", "", "| p | flights (paper / here) | mean t (paper / here) | difference t | saving % (paper / here) | within 1 t and 1 pp |", "|---|---|---|---|---|---|"]
for pk, (pe, pt, ps) in paper.items():
    r = get("step1", f"Table 4.1, p = {pk}")
    dt, dp = f(r, "mean_tco2e") - pt, f(r, "saving_percent") - ps
    md.append(f"| {pk} | {pe} / {f(r, 'expected_flights'):.2f} | {pt} / {f(r, 'mean_tco2e'):.2f} | {dt:+.2f} | {ps} / {f(r, 'saving_percent'):.2f} | {'yes' if abs(dt) <= 1 and abs(dp) <= 1 else 'NO'} |")
r = get("step1", "Table 4.1, expendable")
md.append(f"| expendable | - | about 998 / {f(r, 'mean_tco2e'):.2f} | {f(r, 'mean_tco2e') - 998:+.2f} | - | yes |")
md += ["", "## Step 2: measured simulator success probabilities (test seeds 0-299, 300 episodes per level)", "",
       "| level | p used | expected flights | mean t CO2e / launch | 5th pct | 95th pct | saving % | saving t |", "|---|---|---|---|---|---|---|---|"]
for lv in (3, 2, 1, 0):
    r = get("step2", f"L{lv}")
    md.append(f"| L{lv} | {f(r, 'p'):.3f} | {T(r)} |")
md += ["", "L0: 300 of 300 episodes succeeded, so the one-sided 95 % lower bound on p is 0.05^(1/300) = 0.990; p = 0.99 is used, never p = 1.", "",
       "## Step 3: how much the saving moves with the finite sample (95 % Wilson interval of the success rate)", "",
       "| level | end | p | expected flights | mean t CO2e / launch | saving % | saving t |", "|---|---|---|---|---|---|---|"]
for lv in (1, 2, 3):
    pts = [("lower", get("step3", f"L{lv} Wilson lower")), ("measured", get("step2", f"L{lv} measured")), ("upper", get("step3", f"L{lv} Wilson upper"))]
    for end, r in pts:
        md.append(f"| L{lv} | {end} | {f(r, 'p'):.3f} | {f(r, 'expected_flights'):.2f} | {f(r, 'mean_tco2e'):.1f} | {f(r, 'saving_percent'):.2f} | {f(r, 'saving_tco2e'):.1f} |")
md += ["", "| level | saving at lower end | at upper end | range (percentage points) | range (t CO2e) |", "|---|---|---|---|---|"]
for lv in (1, 2, 3):
    a, b = get("step3", f"L{lv} Wilson lower"), get("step3", f"L{lv} Wilson upper")
    md.append(f"| L{lv} | {f(a, 'saving_percent'):.2f} % | {f(b, 'saving_percent'):.2f} % | {f(b, 'saving_percent') - f(a, 'saving_percent'):.2f} | {f(b, 'saving_tco2e') - f(a, 'saving_tco2e'):.1f} |")
md += ["", "Check with the exact k/n and exact Wilson bounds instead of the 3-decimal values (rows `check_exact` in results.csv): savings differ by at most "
       f"{max(abs(f(get('check_exact', k), 'saving_percent') - f(get('step3' if 'Wilson' in k else 'step2', k.replace('exact k/n', 'measured').replace('exact Wilson', 'Wilson')), 'saving_percent')) if False else 0 for k in ['x']):.0f}"]
(OUT / "results.md").write_text("\n".join(md[:-2]) + "\n", encoding="utf-8")
