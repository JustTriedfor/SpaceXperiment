"""Why does the scripted expert fail at L2 and L3?  (read only: the expert and the simulator are not changed)

    python paper_assets/expert_failures/analyze.py

Inputs: paper_assets/before/episodes.csv (expert_tuned rows, test seeds 0-299) and, for each of those episodes, the start state that env.reset(seed) produces
(initial offset from the pad, velocities, wind mean) - nothing is flown here.  Output: failure_table.md, failure_by_start.csv, printed analysis.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.sim.env import SpaceXEnv  # noqa: E402

OUT = Path(__file__).resolve().parent
rows = [r for r in csv.DictReader(open(ROOT / "paper_assets" / "before" / "episodes.csv")) if r["controller"] == "expert_tuned"]
starts = {}
for lv in range(4):
    env = SpaceXEnv(level=lv)
    env.set_rules_stage(2)
    for seed in range(300):
        env.reset(seed=seed)
        s = env.s
        starts[(lv, seed)] = dict(x0=float(s.x), y0=float(s.y), vx0=float(s.vx), vy0=float(s.vy), wind=float(env.wind.mean), amp=float(env.wind.amp))
data = []
for r in rows:
    lv, seed = int(r["level"]), int(r["seed"])
    d = dict(starts[(lv, seed)])
    d.update(level=lv, seed=seed, success=int(r["success"]), reason=r["failure_reason"], flags=set(filter(None, r["failure_flags"].split("|"))),
             speed=float(r["touchdown_speed_mps"]) if r["touchdown_speed_mps"] else np.nan, offset=float(r["pad_offset_m"]) if r["pad_offset_m"] else np.nan,
             tilt=float(r["tilt_rad"]) if r["tilt_rad"] else np.nan, fuel=float(r["fuel_remaining_kg"]), final_x=float(r["final_x_m"]) if r["final_x_m"] else np.nan)
    data.append(d)
md = ["# Scripted expert: why it fails (test seeds 0-299, 300 episodes per level, noise 0)", "",
      "## 1. Failure reasons per level (one label per failed episode: first match of out_of_fuel > too_fast > tilted > off_pad > crashed_other)", "",
      "| level | episodes | failures | too_fast | tilted | off_pad | out_of_fuel | crashed_other |", "|---|---|---|---|---|---|---|---|"]
for lv in range(4):
    ds = [d for d in data if d["level"] == lv]
    f = [d for d in ds if not d["success"]]
    c = lambda k: sum(1 for d in f if d["reason"] == k)
    md.append(f"| L{lv} | {len(ds)} | {len(f)} | {c('too_fast')} | {c('tilted')} | {c('off_pad')} | {c('out_of_fuel')} | {c('crashed_other')} |")
md += ["", "## 2. Every rule that was broken (an episode can break several; counts of FAILED episodes that broke each rule)", "",
       "| level | failures | off_pad | too_fast | tilted | not_at_rest | out_of_fuel | off_pad AND too_fast |", "|---|---|---|---|---|---|---|---|"]
for lv in range(4):
    f = [d for d in data if d["level"] == lv and not d["success"]]
    n = lambda k: sum(1 for d in f if k in d["flags"])
    md.append(f"| L{lv} | {len(f)} | {n('off_pad')} | {n('too_fast')} | {n('tilted')} | {n('not_at_rest')} | {n('out_of_fuel')} | {sum(1 for d in f if 'off_pad' in d['flags'] and 'too_fast' in d['flags'])} |")
md += ["", "## 3. Size of the failures", "", "| level | outcome | n | median pad offset (m) | median touchdown speed (m/s) | median fuel left (kg) |", "|---|---|---|---|---|---|"]
for lv in (1, 2, 3):
    for name, sel in (("success", lambda d: d["success"]), ("failed", lambda d: not d["success"])):
        ds = [d for d in data if d["level"] == lv and sel(d)]
        md.append(f"| L{lv} | {name} | {len(ds)} | {np.nanmedian([d['offset'] for d in ds]):.3f} | {np.nanmedian([d['speed'] for d in ds]):.3f} | {np.median([d['fuel'] for d in ds]):.3f} |")
md += ["", "## 4. Success rate by how far the start is from the pad, and by wind (L1-L3 pooled per level; terciles of the quantity within each level)", "",
       "| level | quantity | lowest third | middle third | highest third |", "|---|---|---|---|---|"]
for lv in (1, 2, 3):
    ds = [d for d in data if d["level"] == lv]
    for q, f in (("|start x offset|", lambda d: abs(d["x0"])), ("|wind mean|", lambda d: abs(d["wind"])), ("|start vx|", lambda d: abs(d["vx0"])), ("start vy (fall speed)", lambda d: -d["vy0"])):
        v = np.array([f(d) for d in ds])
        lo, hi = np.quantile(v, [1 / 3, 2 / 3])
        parts = [[d for d, x in zip(ds, v) if x <= lo], [d for d, x in zip(ds, v) if lo < x <= hi], [d for d, x in zip(ds, v) if x > hi]]
        md.append(f"| L{lv} | {q} (cuts {lo:.1f} / {hi:.1f}) | " + " | ".join(f"{np.mean([d['success'] for d in p]):.2f} (n={len(p)})" for p in parts) + " |")
md += ["", "## 5. Direction of the miss", ""]
for lv in (1, 2, 3):
    f = [d for d in data if d["level"] == lv and not d["success"] and not np.isnan(d["final_x"])]
    same = np.mean([np.sign(d["final_x"]) == np.sign(d["wind"]) for d in f])
    sx = np.mean([np.sign(d["final_x"]) == np.sign(d["x0"]) for d in f])
    md.append(f"- L{lv}: of {len(f)} failures that touched down, the landing point is on the same side of the pad as the wind push in {same:.2f} and on the same side as the start offset in {sx:.2f}; median |final x| {np.median([abs(d['final_x']) for d in f]):.1f} m.")
(OUT / "failure_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
with open(OUT / "failure_by_start.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["level", "seed", "success", "failure_reason", "x0_m", "y0_m", "vx0_mps", "vy0_mps", "wind_mean_mps", "pad_offset_m", "touchdown_speed_mps", "fuel_remaining_kg"])
    for d in data:
        w.writerow([d["level"], d["seed"], d["success"], d["reason"], d["x0"], d["y0"], d["vx0"], d["vy0"], d["wind"], d["offset"], d["speed"], d["fuel"]])
print("\n".join(md))
