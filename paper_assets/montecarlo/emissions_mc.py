"""Emissions Monte Carlo REBUILT from Chapter 3 of the paper (the original script could not be found on this machine).

    python paper_assets/montecarlo/emissions_mc.py                (all steps; writes results.csv, results.md, fig4_3_emissions_vs_p.png)
    python paper_assets/montecarlo/emissions_mc.py --step1        (only the reproduction of Table 4.1)

WHAT IS TAKEN FROM THE TEXT (Chapter 3, Table 3.1)                                               WHAT IS NOT IN THE TEXT (inferred, then checked against Table 4.1)
  draws                200,000                                                                      the closed-form formulas (they are in "Section 2.5", which was not pasted)
  P1 first stage       330 .. 500, most likely 423.5 t CO2e                                         the distribution type (the text gives min / most likely / max: triangular)
  P2 second stage      68 .. 110, most likely 90                                                    the random seed (not stated)
  F  propellant        430 .. 520, most likely 470                                                  integer or continuous N, the percentile method
  T  transport         10 .. 30, most likely 14
  r  refurbishment     2 .. 10 percent of P1 per reflight, most likely 5
  N  max flights       15 .. 40, uniform

FORMULAS (inferred; they reproduce all 6 rows of Table 4.1 - see README.md)
  expected flights of one booster   E = (1 - p^N) / (1 - p)      (it flies until the first failed landing or N flights; E = N when p = 1)
  expendable launch                 X = P1 + P2 + F + T
  reusable launch, per launch       C = P1 * (1 + r * (E - 1)) / E + P2 + F + T      (the first stage is made once per E flights, refurbished E-1 times, r*P1 each)
All inputs are drawn ONCE with one seed and every p is evaluated on the SAME draws (common random numbers), so rows differ only because of p.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DRAWS = 200_000                      # from the text
SEED = 20260101                      # NOT in the text: chosen here, fixed for every p
INPUTS = dict(P1=(330.0, 423.5, 500.0), P2=(68.0, 90.0, 110.0), F=(430.0, 470.0, 520.0), T=(10.0, 14.0, 30.0), r_percent=(2.0, 5.0, 10.0))   # (min, most likely, max) from Table 3.1
N_RANGE = (15, 40)                   # uniform, from Table 3.1 (integer flights, both ends included)
PERCENTILES = (5, 95)


def draw(n=DRAWS, seed=SEED, n_integer=True):
    rng = np.random.default_rng(seed)
    d = {k: rng.triangular(lo, mode, hi, n) for k, (lo, mode, hi) in INPUTS.items()}
    d["r"] = d.pop("r_percent") / 100.0
    d["N"] = rng.integers(N_RANGE[0], N_RANGE[1] + 1, n).astype(float) if n_integer else rng.uniform(N_RANGE[0], N_RANGE[1], n)
    return d


def flights(p, N):
    return N.copy() if p >= 1.0 else (1.0 - p ** N) / (1.0 - p)


def per_launch(d, p):
    E = flights(p, d["N"])
    return d["P1"] * (1.0 + d["r"] * (E - 1.0)) / E + d["P2"] + d["F"] + d["T"], E


def row(d, label, p, expendable):
    """expendable = per-draw X array (the same draws)."""
    if p is None:
        c, E = expendable, np.ones_like(expendable)
    else:
        c, E = per_launch(d, p)
    mean_x = float(expendable.mean())
    m = float(c.mean())
    return dict(label=label, p="" if p is None else repr(float(p)), expected_flights=repr(float(E.mean())), mean_tco2e=repr(m), p5_tco2e=repr(float(np.percentile(c, PERCENTILES[0]))),
                p95_tco2e=repr(float(np.percentile(c, PERCENTILES[1]))), expendable_mean_tco2e=repr(mean_x), saving_tco2e=repr(mean_x - m), saving_percent=repr(100.0 * (mean_x - m) / mean_x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1", action="store_true")
    ap.add_argument("--continuous-n", action="store_true", help="draw N as a continuous uniform number instead of an integer (a check; the default is integer)")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    d = draw(seed=a.seed, n_integer=not a.continuous_n)
    X = d["P1"] + d["P2"] + d["F"] + d["T"]
    rows = []
    step1 = [("step1", f"Table 4.1, p = {p}", p) for p in (0.33, 0.60, 0.90, 0.95, 0.99, 0.9963)]
    for step, label, p in step1:
        rows.append(dict(step=step, **row(d, label, p, X)))
    rows.append(dict(step="step1", **row(d, "Table 4.1, expendable", None, X)))
    if not a.step1:
        for lv, p, note in ((3, 0.300, "L3"), (2, 0.537, "L2"), (1, 0.863, "L1"), (0, 0.99, "L0 (300 of 300: one-sided 95 % lower bound 0.990, not p = 1)")):
            rows.append(dict(step="step2", **row(d, f"{note} measured p = {p}", p, X)))
        for lv, lo, hi in ((1, 0.820, 0.898), (2, 0.480, 0.592), (3, 0.251, 0.354)):
            rows.append(dict(step="step3", **row(d, f"L{lv} Wilson lower p = {lo}", lo, X)))
            rows.append(dict(step="step3", **row(d, f"L{lv} Wilson upper p = {hi}", hi, X)))
        # the same with the exact k/n and exact Wilson bounds (retrain/summary.csv), to show what rounding the inputs to 3 decimals does
        import csv as _csv
        for r in _csv.DictReader(open(ROOT / "paper_assets" / "retrain" / "summary.csv")):
            if r["controller"] == "sx_retrained" and int(r["level"]) in (1, 2, 3):
                lv = int(r["level"])
                for tag, p in (("exact k/n", float(r["success_rate"])), ("exact Wilson lower", float(r["wilson_lower"])), ("exact Wilson upper", float(r["wilson_upper"]))):
                    rows.append(dict(step="check_exact", **row(d, f"L{lv} {tag} p = {p!r}", p, X)))
    cols = list(rows[0].keys())
    path = OUT / ("results_step1_only.csv" if a.step1 else "results.csv")
    if a.continuous_n:
        path = OUT / "results_check_continuous_n.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, "| seed", a.seed, "| draws", DRAWS, "| script sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16])
    for r in rows:
        print(f"{r['step']:11s} {r['label'][:58]:58s} E={float(r['expected_flights']):6.2f}  mean={float(r['mean_tco2e']):8.2f}  p5={float(r['p5_tco2e']):8.2f}  p95={float(r['p95_tco2e']):8.2f}  saving={float(r['saving_percent']):6.2f} % ({float(r['saving_tco2e']):6.2f} t)")


if __name__ == "__main__":
    main()
