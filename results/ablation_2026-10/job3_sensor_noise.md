# Job 3: sensor noise on the observations (trained controller, not retrained, seeds 0-299, 300 episodes per cell)

Gaussian noise added to every observation at every step, standard deviation = noise % of that observation's range (range measured on clean flights, validation seeds; see `data/obs_ranges.json`). Cells: successes / 300, success rate [95 % Wilson interval].

| noise | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| 0 % | 300/300 = 1.000 [0.987, 1.000] | 259/300 = 0.863 [0.820, 0.898] | 161/300 = 0.537 [0.480, 0.592] | 90/300 = 0.300 [0.251, 0.354] |
| 1 % | 0/300 = 0.000 [0.000, 0.013] | 0/300 = 0.000 [0.000, 0.013] | 0/300 = 0.000 [0.000, 0.013] | 0/300 = 0.000 [0.000, 0.013] |
| 2 % | 0/300 = 0.000 [0.000, 0.013] | 0/300 = 0.000 [0.000, 0.013] | 0/300 = 0.000 [0.000, 0.013] | 0/300 = 0.000 [0.000, 0.013] |
| 5 % | 0/300 = 0.000 [0.000, 0.013] | 0/300 = 0.000 [0.000, 0.013] | 0/300 = 0.000 [0.000, 0.013] | 0/300 = 0.000 [0.000, 0.013] |

Noise 0 % is the clean run: 300, 259, 161, 90 successes (the paper: 300, 259, 161, 90).

Change in success rate against 0 % (percentage points):

| noise | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| 1 % | -100.0 | -86.3 | -53.7 | -30.0 |
| 2 % | -100.0 | -86.3 | -53.7 | -30.0 |
| 5 % | -100.0 | -86.3 | -53.7 | -30.0 |
