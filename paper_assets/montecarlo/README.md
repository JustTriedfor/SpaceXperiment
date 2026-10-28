# Emissions Monte Carlo at the simulator's measured success probabilities

## Read this first: the original script was not found, so the model was rebuilt

I searched this PC (C: and D:, the project folders, OneDrive, Downloads, earlier Claude session logs) and did not find the script that produced Table 4.1. **Nothing was copied.** `emissions_mc.py` is a rebuild from Chapter 3 of the paper. The chapter gives the inputs (Table 3.1) and the draw count, but not the formulas (they are in "Section 2.5", which was not provided), not the distribution type, and not the random seed. Those were inferred, and the check is Step 1: the rebuild reproduces all six rows of Table 4.1. That supports the formulas but does not prove they are the original ones.

| item | value | source |
|---|---|---|
| draws | 200,000 | **text** (Section 3.2) |
| P1 first stage | 330 to 500, most likely 423.5 t CO2e | **text** (Table 3.1) |
| P2 second stage | 68 to 110, most likely 90 | **text** |
| F propellant combustion | 430 to 520, most likely 470 | **text** |
| T transport | 10 to 30, most likely 14 | **text** |
| r refurbishment per reflight | 2 to 10 percent of P1, most likely 5 | **text** |
| N maximum flights per booster | 15 to 40, uniform | **text** ("uniform"); integer flights, both ends included: **inferred** |
| distribution of P1, P2, F, T, r | triangular (min, most likely, max) | **inferred** (the text gives only a range and a most likely value) |
| expected flights | E = (1 - p^N) / (1 - p); E = N at p = 1 | **inferred** |
| expendable launch | X = P1 + P2 + F + T | **inferred** (mean 998.55 t = the paper's "about 998") |
| reusable launch | C = P1 (1 + r (E - 1)) / E + P2 + F + T | **inferred** |
| seed | 20260101 | **my choice** (the original seed is unknown); the same draws are used for every p |
| percentiles | 5th and 95th percentile of C over the draws (numpy default) | **inferred** |

The seed does not matter at this precision: seeds 1, 2, 3 and 20260101 give means within 0.2 t of each other (for example 620.60 to 620.73 t at p = 0.9963). Drawing N as a continuous number instead of an integer changes means by at most 0.1 t.

Files: `emissions_mc.py` (SHA-256 `6bd18160ac6f123360b2d1a1dca0498c8fa6b0c80fafbacf798c6e31a6aa7a9c`), `make_outputs.py` (`4e5ca085ea1589e84b4319929038a53d533927bb8bb3366d5a34a830a62f080a`), `results.csv` (`358034b390652fbc99426a9f3e7859debb71d7fe179585240a03d447e9b69ae8`), `results.md`, `fig4_3_emissions_vs_p.png` (1600 x 900), `results_check_continuous_n.csv`.

```bash
python paper_assets/montecarlo/emissions_mc.py                 # results.csv (all rows, full precision); about 2 s
python paper_assets/montecarlo/emissions_mc.py --step1 --continuous-n   # check with continuous N (results_check_continuous_n.csv)
python paper_assets/montecarlo/make_outputs.py                 # fig4_3_emissions_vs_p.png and results.md
```

## Step 1: reproduction of Table 4.1 (passed: every row within 1 t and 1 percentage point)

| p | expected flights, paper / here | mean t CO2e, paper / here | difference (t) | saving %, paper / here |
|---|---|---|---|---|
| 0.33 | 1.5 / 1.49 | 868 / 868.45 | +0.45 | 13 / 13.03 |
| 0.60 | 2.5 / 2.50 | 762 / 762.02 | +0.02 | 24 / 23.69 |
| 0.90 | 9.3 / 9.26 | 647 / 647.06 | +0.06 | 35 / 35.20 |
| 0.95 | 14.8 / 14.76 | 631 / 631.58 | +0.58 | 37 / 36.75 |
| 0.99 | 24.0 / 23.95 | 622 / 621.87 | -0.13 | 38 / 37.72 |
| 0.9963 | 26.1 / 26.12 | 620 / 620.60 | +0.60 | 38 / 37.85 |
| expendable | about 998 / 998.55 | | +0.55 | |

Three rows (0.95, 0.9963 and the expendable case) are 0.55 to 0.60 t above the paper's integers, and the paper's 622 at p = 0.99 is rounded while 631 and 620 look truncated. Monte Carlo error with 200,000 draws is only about 0.05 to 0.1 t, and neither seed nor integer N moves it enough, so the original model must differ from my rebuild in some small detail that I cannot identify without it. The differences are inside your tolerance, so I went on; please keep this in mind if Table 4.1 is to be reproduced to the tonne.

## Steps 2 and 3: results (means over 200,000 draws; the full table is `results.md`)

| level | p used | expected flights | mean t CO2e per launch | 5th to 95th percentile | saving vs expendable (998.5 t) |
|---|---|---|---|---|---|
| L3 | 0.300 | 1.43 | 880.3 | 826.2 to 933.5 | 11.84 % (118.3 t) |
| L2 | 0.537 | 2.16 | 786.9 | 742.2 to 831.7 | 21.20 % (211.7 t) |
| L1 | 0.863 | 7.08 | 660.1 | 623.5 to 698.0 | 33.90 % (338.5 t) |
| L0 | 0.990 | 23.95 | 621.9 | 585.5 to 659.5 | 37.72 % (376.7 t) |

L0 succeeded in 300 of 300 episodes, so p = 1 is not used: the one-sided 95 % lower bound on p is 0.05^(1/300) = 0.990, and p = 0.99 is used as a lower bound, not a measurement. The levels' p are the retrained single-lander controller's test results (`paper_assets/retrain/README.md`).

Effect of the finite sample (95 % Wilson interval of the success rate, at both ends):

| level | p range | saving at lower / upper end | range of the saving | mean t CO2e at lower / upper end |
|---|---|---|---|---|
| L1 | 0.820 to 0.898 | 32.29 % / 35.13 % | 2.84 points (28.3 t) | 676.1 / 647.7 |
| L2 | 0.480 to 0.592 | 18.95 % / 23.37 % | 4.42 points (44.1 t) | 809.3 / 765.2 |
| L3 | 0.251 to 0.354 | 9.91 % / 13.98 % | 4.07 points (40.6 t) | 899.6 / 859.0 |

Using the exact k/n and exact Wilson bounds instead of the rounded 3-decimal values (rows `check_exact`) changes the savings by at most 0.013 percentage points.

## Anything surprising

- The saving is steeply nonlinear at low p. Between L2 and L1 (p from 0.537 to 0.863) it rises by 12.7 points; between L1 and L0 (p 0.863 to 0.99) it rises only 3.8. So the extra landing reliability above about 0.86 buys little in emissions in this model, while the low-p cases are where the sample size matters (4 points of saving).
- Even at L3 (p = 0.300, expected 1.43 flights) the model still gives a saving of about 12 %, because the first stage is amortised over more than one flight on average. The 5th to 95th percentile band at every p stays below the expendable mean except for p close to 0 (the band top reaches 933 t at p = 0.3 and 1,064 t at p = 0), so the saving is not significant per draw at low p; the table compares means.
- In the inferred formulas a failed landing only ends the booster's life (the next launch needs a new first stage), and the second stage, propellant and transport are the same for reusable and expendable launches. If Section 2.5 treats either differently, these numbers change.
- The measured p are per-landing success rates in the simulator, under its success rule and its levels (wind scale 0 / 5 / 10 / 15 m/s with random per-episode means), not Falcon flight statistics; the Falcon record 0.9963 is the paper's own input.
