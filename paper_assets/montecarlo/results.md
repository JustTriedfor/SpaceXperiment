# Emissions Monte Carlo results (rebuilt model, 200,000 draws, one seed for every p; full precision in results.csv)

Columns: expected flights = mean over the draws (reuse limit N uniform 15 to 40); mean and 5th / 95th percentile of tonnes CO2e per launch; saving against the expendable mean (998.5 t on the same draws). Rounding here: flights 2 decimals, tonnes 1 decimal, percent 2 decimals.

## Step 1: reproduction of Table 4.1

| p | flights (paper / here) | mean t (paper / here) | difference t | saving % (paper / here) | within 1 t and 1 pp |
|---|---|---|---|---|---|
| 0.33 | 1.5 / 1.49 | 868 / 868.45 | +0.45 | 13 / 13.03 | yes |
| 0.6 | 2.5 / 2.50 | 762 / 762.02 | +0.02 | 24 / 23.69 | yes |
| 0.9 | 9.3 / 9.26 | 647 / 647.06 | +0.06 | 35 / 35.20 | yes |
| 0.95 | 14.8 / 14.76 | 631 / 631.58 | +0.58 | 37 / 36.75 | yes |
| 0.99 | 24.0 / 23.95 | 622 / 621.87 | -0.13 | 38 / 37.72 | yes |
| 0.9963 | 26.1 / 26.12 | 620 / 620.60 | +0.60 | 38 / 37.85 | yes |
| expendable | - | about 998 / 998.55 | +0.55 | - | yes |

## Step 2: measured simulator success probabilities (test seeds 0-299, 300 episodes per level)

| level | p used | expected flights | mean t CO2e / launch | 5th pct | 95th pct | saving % | saving t |
|---|---|---|---|---|---|---|---|
| L3 | 0.300 | 1.43 | 880.3 | 826.2 | 933.5 | 11.84 | 118.3 |
| L2 | 0.537 | 2.16 | 786.9 | 742.2 | 831.7 | 21.20 | 211.7 |
| L1 | 0.863 | 7.08 | 660.1 | 623.5 | 698.0 | 33.90 | 338.5 |
| L0 | 0.990 | 23.95 | 621.9 | 585.5 | 659.5 | 37.72 | 376.7 |

L0: 300 of 300 episodes succeeded, so the one-sided 95 % lower bound on p is 0.05^(1/300) = 0.990; p = 0.99 is used, never p = 1.

## Step 3: how much the saving moves with the finite sample (95 % Wilson interval of the success rate)

| level | end | p | expected flights | mean t CO2e / launch | saving % | saving t |
|---|---|---|---|---|---|---|
| L1 | lower | 0.820 | 5.50 | 676.1 | 32.29 | 322.5 |
| L1 | measured | 0.863 | 7.08 | 660.1 | 33.90 | 338.5 |
| L1 | upper | 0.898 | 9.11 | 647.7 | 35.13 | 350.8 |
| L2 | lower | 0.480 | 1.92 | 809.3 | 18.95 | 189.2 |
| L2 | measured | 0.537 | 2.16 | 786.9 | 21.20 | 211.7 |
| L2 | upper | 0.592 | 2.45 | 765.2 | 23.37 | 233.4 |
| L3 | lower | 0.251 | 1.34 | 899.6 | 9.91 | 99.0 |
| L3 | measured | 0.300 | 1.43 | 880.3 | 11.84 | 118.3 |
| L3 | upper | 0.354 | 1.55 | 859.0 | 13.98 | 139.6 |

| level | saving at lower end | at upper end | range (percentage points) | range (t CO2e) |
|---|---|---|---|---|
| L1 | 32.29 % | 35.13 % | 2.84 | 28.3 |
| L2 | 18.95 % | 23.37 % | 4.42 | 44.1 |
| L3 | 9.91 % | 13.98 % | 4.07 | 40.6 |
