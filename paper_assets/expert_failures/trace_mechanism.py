"""Does the scripted expert fail at L2 and L3 because the sideways error is already too large when the burn starts?  (new script; the expert and simulator are not changed)

    python paper_assets/expert_failures/trace_mechanism.py [workers]

Flies the expert on the paper's test seeds 0-299 at L1, L2 and L3 with the same loop as paper_assets/scripts/run_eval.py (controller expert_tuned, noise 0), and logs, for
every control step, the height, sideways position and speed, the tilt the expert asked for and the tilt cap it is limited to.  The expert code is only read.
Writes trace_mechanism.csv (one row per episode) and trace_mechanism.md (the tables).  Association from the ignition state, not an intervention: see the md for what it can and cannot show.
"""
import csv
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
LEVELS = (1, 2, 3)
SEEDS = range(300)


def fly(args):
    level, seed = args
    from src.sim import expert as E
    from src.sim.env import SpaceXEnv
    from src.sim.expert_data import _label
    env = SpaceXEnv(level=level)
    env.set_rules_stage(2)
    env.reset(seed=seed)
    s0 = env.s
    start = dict(x0=float(s0.x), vx0=float(s0.vx), vd0=float(-s0.vy), wind=float(env.wind.mean))
    ex = E.Expert()
    ex.reset()
    rows = []
    done = False
    while not done:
        s = env.s
        cmd = _label(ex, env, False)                       # the expert decides on the state before the step (this call is what the paper's run does)
        h = max(float(s.foot_height()), 0.0)
        cap = min(E.LATERAL_TILT_MAX, E.LATERAL_TILT_GROUND + E.LATERAL_TILT_PER_M * h)
        acc_cap = max(float(s.thrust_n), 0.0) / float(s.mass) * math.sin(cap)           # sideways acceleration the thrust could give with the nose tilted to the cap
        rows.append((h, float(s.x), float(s.vx), max(-float(s.vy), 0.0), bool(ex.burning), float(ex.dbg[0]), cap, acc_cap, float(s.th), abs(float(cmd[1])), abs(float(cmd[2]))))
        _, _, te, tr, info = env.step(cmd)
        done = te or tr
    m = info["episode_metrics"]
    r = np.array(rows, dtype=float)
    burn = r[:, 4] > 0.5
    out = dict(level=level, seed=seed, success=int(bool(m["success"])), dist_m=float(m["dist_m"]), speed_mps=float(m["speed_mps"]), end=str(m["end"]), **start)
    if burn.any():
        i = int(np.argmax(burn))
        h, x, vx, vd = r[i, 0], r[i, 1], r[i, 2], r[i, 3]
        t_go = 2.0 * h / (vd + E.V_TOUCH)
        sat = burn & (np.abs(r[:, 5]) >= r[:, 6] - 1e-9)
        sat_idx = np.where(sat)[0]
        a_poly = abs(6.0 * x / t_go ** 2 + 4.0 * vx / t_go)                      # sideways acceleration (m/s^2, constant) that puts x and vx at zero together at t_go
        cap_acc = float(r[burn & (r[:, 0] > 30.0), 7].mean()) if (burn & (r[:, 0] > 30.0)).any() else float("nan")
        hi = burn & (r[:, 0] > 150.0) & (np.abs(r[:, 5]) > 0.2)               # burn steps above 150 m where the expert asks for a real tilt (more than 0.2 rad)
        deliv = float(np.median(np.clip(r[hi, 8] * np.sign(r[hi, 5]) / np.abs(r[hi, 5]), -1.0, 1.0))) if hi.any() else float("nan")
        hb = burn & (r[:, 0] > 150.0)
        gim_sat = float((r[hb, 9] >= 0.999).mean()) if hb.any() else float("nan")
        fin_sat = float((r[hb, 10] >= 0.999).mean()) if hb.any() else float("nan")
        j100 = int(np.argmax(r[:, 0] <= 100.0)) if (r[:, 0] <= 100.0).any() else len(r) - 1
        out.update(burned=1, h_ig=h, x_ig=x, vx_ig=vx, vd_ig=vd, t_go_ig=t_go, zem_ig=x + vx * t_go, deliv=deliv, gim_sat=gim_sat, fin_sat=fin_sat, a_poly_ig=a_poly, cap_acc=cap_acc, demand=a_poly / cap_acc if cap_acc > 0 else float("nan"), burn_steps=int(burn.sum()), sat_frac=float(sat.sum() / burn.sum()),
                   first_sat_h=float(r[sat_idx[0], 0]) if len(sat_idx) else float("nan"), sat_below_30=int(bool(sat[r[:, 0] < 30.0].any())),
                   x_100=float(r[j100, 1]), vx_100=float(r[j100, 2]))
    else:
        out.update(burned=0)
    return out


def auc(pos, neg):
    """probability that a random failure has a larger value than a random success (0.5 = no information); ties count half"""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return gt / (len(pos) * len(neg))


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    jobs = [(lv, sd) for lv in LEVELS for sd in SEEDS]
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(fly, jobs, chunksize=3)
    print(f"flown {len(res)} episodes in {time.time() - t0:.0f} s")
    cols = ["level", "seed", "success", "dist_m", "speed_mps", "end", "x0", "vx0", "vd0", "wind", "burned", "h_ig", "x_ig", "vx_ig", "vd_ig", "t_go_ig", "zem_ig", "deliv", "gim_sat", "fin_sat", "a_poly_ig", "cap_acc", "demand", "burn_steps", "sat_frac",
            "first_sat_h", "sat_below_30", "x_100", "vx_100"]
    with open(OUT / "trace_mechanism.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for d in res:
            w.writerow([repr(float(d[c])) if isinstance(d.get(c, ""), float) else d.get(c, "") for c in cols])
    md = ["# Is the expert's miss decided before the burn?  (test seeds 0-299, expert_tuned, noise 0; flown again with a per-step log, nothing in the expert or simulator changed)", "",
          "Ignition state = height, sideways position x and sideways speed vx at the first control step with the engine latched on. `zem` = x + vx * t_go with t_go = 2h/(fall speed + V_TOUCH): where the stage would touch down relative to the pad if nothing were done sideways from then on. `demand` = the constant sideways acceleration that would bring x and vx to zero together at t_go, divided by the sideways acceleration the thrust could give with the tilt at its cap (average over the burn above 30 m); above 1 the tilt cap alone makes the miss unavoidable. `sat` = the tilt the expert asks for is at its cap (min(LATERAL_TILT_MAX, GROUND + PER_M*h)).", ""]
    ok_total = {lv: sum(d["success"] for d in res if d["level"] == lv) for lv in LEVELS}
    md.append("Check against the paper: successes " + ", ".join(f"L{lv} {ok_total[lv]}/300" for lv in LEVELS) + " (paper: expert_tuned 258 / 161 / 95).")
    md += ["", "## 1. Ignition state of successes and failures (medians)", "", "| level | outcome | n | ignition height (m) | abs x at ignition (m) | abs vx (m/s) | abs zem (m) | share of burn with tilt at cap | cap hit below 30 m |", "|---|---|---|---|---|---|---|---|---|"]
    for lv in LEVELS:
        for name, sel in (("success", 1), ("failed", 0)):
            ds = [d for d in res if d["level"] == lv and d["success"] == sel and d["burned"]]
            md.append(f"| L{lv} | {name} | {len(ds)} | {np.median([d['h_ig'] for d in ds]):.0f} | {np.median([abs(d['x_ig']) for d in ds]):.1f} | {np.median([abs(d['vx_ig']) for d in ds]):.1f} | "
                      f"{np.median([abs(d['zem_ig']) for d in ds]):.1f} | {np.median([d['sat_frac'] for d in ds]):.2f} | {np.mean([d['sat_below_30'] for d in ds]):.2f} |")
    md += ["", "## 2. How well does each quantity tell failures from successes?  (AUC: 0.5 = no information, 1.0 = every failure ranks above every success)", "",
           "| level | demand / capacity | abs zem at ignition | abs x at ignition | abs vx at ignition | abs start x | abs wind mean | abs x at 100 m | abs vx at 100 m | share of burn at cap |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lv in LEVELS:
        ds = [d for d in res if d["level"] == lv and d["burned"]]
        f_, s_ = [d for d in ds if not d["success"]], [d for d in ds if d["success"]]
        g = lambda k, a=abs: ([a(d[k]) for d in f_], [a(d[k]) for d in s_])
        md.append(f"| L{lv} | " + " | ".join(f"{auc(*g(k)):.2f}" for k in ("demand", "zem_ig", "x_ig", "vx_ig", "x0", "wind", "x_100", "vx_100")) + f" | {auc(*g('sat_frac', lambda v: v)):.2f} |")
    md += ["", "## 3. Success rate by demand / capacity at ignition", "", "| demand / capacity | L1 | L2 | L3 |", "|---|---|---|---|"]
    edges = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 1e9]
    for a, b in zip(edges[:-1], edges[1:]):
        cells = []
        for lv in LEVELS:
            ds = [d for d in res if d["level"] == lv and d["burned"] and a <= d["demand"] < b]
            cells.append(f"{np.mean([d['success'] for d in ds]):.2f} (n={len(ds)})" if ds else "-")
        md.append(f"| {a:g} to {'inf' if b > 1e8 else format(b, 'g')} | " + " | ".join(cells) + " |")
    md += ["", "Note: the zero-effort miss alone does not separate the groups (its AUC is below 0.5): failures often start with a large sideways speed already pointing at the pad, so x + vx*t_go is small, but stopping that speed is what costs tilt. The demand / capacity ratio includes both."]
    md += ["", "## 4. Does the steering run out of authority, and where?  (burn above 150 m, steps where the expert asks for more than 0.2 rad of tilt)", "",
           "| level | outcome | tilt at cap for more than half of the burn | median height of first cap hit (m) | tilt delivered / tilt asked (median) | gimbal at its limit | fins at their limit |", "|---|---|---|---|---|---|---|"]
    for lv in LEVELS:
        for name, sel in (("success", 1), ("failed", 0)):
            ds = [d for d in res if d["level"] == lv and d["success"] == sel and d["burned"]]
            fh = [d["first_sat_h"] for d in ds if not math.isnan(d["first_sat_h"])]
            md.append(f"| L{lv} | {name} | {np.mean([d['sat_frac'] > 0.5 for d in ds]):.2f} | {np.median(fh) if fh else float('nan'):.0f} | {np.nanmedian([d['deliv'] for d in ds]):.2f} | "
                      f"{np.nanmedian([d['gim_sat'] for d in ds]):.2f} | {np.nanmedian([d['fin_sat'] for d in ds]):.2f} |")
    md += ["", "AUC of tilt delivered / tilt asked for telling failures from successes: " + ", ".join(f"L{lv} {auc([-d['deliv'] for d in res if d['level'] == lv and d['burned'] and not d['success']], [-d['deliv'] for d in res if d['level'] == lv and d['burned'] and d['success']]):.2f}" for lv in LEVELS) + " (lower delivered tilt in failures = higher AUC)."]
    never = [d for d in res if not d["burned"]]
    md += ["", f"Episodes where the engine never latched on: {len(never)} of {len(res)}."]
    (OUT / "trace_mechanism.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
