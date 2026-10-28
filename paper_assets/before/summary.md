# Summary (95 % Wilson intervals; 3 decimals; full precision in summary.csv)

n counts episodes, except `pair_safe_final` where it counts landers (two per episode). `pair_safe_final (joint)` counts episodes in which both landers succeeded and landed within the sync tolerance.

| controller | level | n | successes | success rate | Wilson lower | Wilson upper | mean speed (m/s) | median speed (m/s) | mean tilt (rad) | mean pad offset (m) | mean fuel left (kg) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sx_imitation_quiet | L0 | 300 | 298 | 0.993 | 0.976 | 0.998 | 2.422 | 2.456 | 0.005 | 1.944 | 2375.754 |
| sx_imitation_quiet | L1 | 300 | 204 | 0.680 | 0.625 | 0.730 | 4.268 | 2.534 | 0.036 | 5.669 | 2237.265 |
| sx_imitation_quiet | L2 | 300 | 113 | 0.377 | 0.324 | 0.433 | 9.114 | 7.226 | 0.064 | 29.366 | 2128.541 |
| sx_imitation_quiet | L3 | 300 | 43 | 0.143 | 0.108 | 0.188 | 14.964 | 14.700 | 0.118 | 88.396 | 2071.047 |
| pair_safe_final | L0 | 600 | 599 | 0.998 | 0.991 | 1.000 | 2.034 | 1.979 | 0.004 | 1.156 | 2137.285 |
| pair_safe_final | L1 | 600 | 525 | 0.875 | 0.846 | 0.899 | 2.152 | 2.086 | 0.005 | 4.133 | 2025.544 |
| pair_safe_final | L2 | 600 | 353 | 0.588 | 0.549 | 0.627 | 2.973 | 2.205 | 0.036 | 21.084 | 1941.047 |
| pair_safe_final | L3 | 600 | 210 | 0.350 | 0.313 | 0.389 | 5.982 | 2.670 | 0.085 | 62.734 | 1885.518 |
| expert_tuned | L0 | 300 | 300 | 1.000 | 0.987 | 1.000 | 2.037 | 2.013 | 0.005 | 1.096 | 2260.431 |
| expert_tuned | L1 | 300 | 258 | 0.860 | 0.816 | 0.895 | 2.178 | 2.117 | 0.006 | 4.825 | 2098.084 |
| expert_tuned | L2 | 300 | 161 | 0.537 | 0.480 | 0.592 | 3.418 | 2.271 | 0.050 | 26.452 | 1975.902 |
| expert_tuned | L3 | 300 | 95 | 0.317 | 0.267 | 0.371 | 7.137 | 3.047 | 0.095 | 75.828 | 1901.021 |
| pair_safe_final (joint) | L0 | 300 | 299 | 0.997 | 0.981 | 0.999 | 1.962 | 1.951 | 0.004 | 1.119 | 2130.483 |
| pair_safe_final (joint) | L1 | 300 | 258 | 0.860 | 0.816 | 0.895 | 2.178 | 2.128 | 0.005 | 4.217 | 2020.045 |
| pair_safe_final (joint) | L2 | 300 | 172 | 0.573 | 0.517 | 0.628 | 3.031 | 2.333 | 0.035 | 21.188 | 1936.689 |
| pair_safe_final (joint) | L3 | 300 | 96 | 0.320 | 0.270 | 0.375 | 5.878 | 2.873 | 0.085 | 64.254 | 1880.474 |

## Failure reasons (counts)

| controller | level | none | too_fast | tilted | off_pad | out_of_fuel | crashed_other |
|---|---|---|---|---|---|---|---|
| sx_imitation_quiet | L0 | 298 | 2 | 0 | 0 | 0 | 0 |
| sx_imitation_quiet | L1 | 204 | 96 | 0 | 0 | 0 | 0 |
| sx_imitation_quiet | L2 | 113 | 186 | 0 | 0 | 0 | 1 |
| sx_imitation_quiet | L3 | 43 | 255 | 0 | 0 | 0 | 2 |
| pair_safe_final | L0 | 599 | 1 | 0 | 0 | 0 | 0 |
| pair_safe_final | L1 | 525 | 9 | 0 | 66 | 0 | 0 |
| pair_safe_final | L2 | 353 | 105 | 0 | 142 | 0 | 0 |
| pair_safe_final | L3 | 210 | 270 | 0 | 120 | 0 | 0 |
| expert_tuned | L0 | 300 | 0 | 0 | 0 | 0 | 0 |
| expert_tuned | L1 | 258 | 2 | 0 | 40 | 0 | 0 |
| expert_tuned | L2 | 161 | 68 | 0 | 71 | 0 | 0 |
| expert_tuned | L3 | 95 | 151 | 0 | 54 | 0 | 0 |

## How many clean trials prove a reliability?

With every trial a success, the one-sided 95 % lower bound on the success probability p is 0.05^(1/n) (exact binomial), or about 1 - 3/n (rule of three).

| claim | exact binomial: clean trials needed | rule of three: clean trials needed |
|---|---|---|
| p >= 0.99 | 299 | 300 |
| p >= 0.9963 | 809 | 811 |

What was achieved (one-sided 95 % Clopper-Pearson lower bound, `clean` = no failure in n trials):

| controller | level | n | failures | clean | one-sided 95 % lower bound on p | supports p >= 0.99 | supports p >= 0.9963 |
|---|---|---|---|---|---|---|---|
| sx_imitation_quiet | L0 | 300 | 2 | no | 0.979 | no | no |
| sx_imitation_quiet | L1 | 300 | 96 | no | 0.633 | no | no |
| sx_imitation_quiet | L2 | 300 | 187 | no | 0.330 | no | no |
| sx_imitation_quiet | L3 | 300 | 257 | no | 0.111 | no | no |
| pair_safe_final | L0 | 600 | 1 | no | 0.992 | yes | no |
| pair_safe_final | L1 | 600 | 75 | no | 0.851 | no | no |
| pair_safe_final | L2 | 600 | 247 | no | 0.554 | no | no |
| pair_safe_final | L3 | 600 | 390 | no | 0.318 | no | no |
| expert_tuned | L0 | 300 | 0 | yes | 0.990 | yes | no |
| expert_tuned | L1 | 300 | 42 | no | 0.823 | no | no |
| expert_tuned | L2 | 300 | 139 | no | 0.488 | no | no |
| expert_tuned | L3 | 300 | 205 | no | 0.272 | no | no |
| pair_safe_final (joint) | L0 | 300 | 1 | no | 0.984 | no | no |
| pair_safe_final (joint) | L1 | 300 | 42 | no | 0.823 | no | no |
| pair_safe_final (joint) | L2 | 300 | 128 | no | 0.524 | no | no |
| pair_safe_final (joint) | L3 | 300 | 204 | no | 0.276 | no | no |

Notes: the two landers of a pair share one wind, one start path and one sync reward, so their outcomes are not independent; the per-lander n for `pair_safe_final` therefore overstates the information (the joint rows, n = episodes, are the honest count). Each episode of the other controllers is an independent draw (seeds 0-299).
