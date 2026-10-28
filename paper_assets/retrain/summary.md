# Retrained single-lander controller (`rt1_final`) on the held-out test seeds 0-299 (95 % Wilson intervals; 3 decimals; full precision in summary.csv)

`sx_retrained` is the new controller; the other rows are the old results copied unchanged from `paper_assets/before/` (same evaluator, same seeds). `pair_safe_final` counts landers (two per episode).

| controller | level | n | successes | success rate | Wilson lower | Wilson upper | mean speed (m/s) | median speed (m/s) | mean tilt (rad) | mean pad offset (m) | mean fuel left (kg) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sx_imitation_quiet | L0 | 300 | 298 | 0.993 | 0.976 | 0.998 | 2.422 | 2.456 | 0.005 | 1.944 | 2375.754 |
| sx_imitation_quiet | L1 | 300 | 204 | 0.680 | 0.625 | 0.730 | 4.268 | 2.534 | 0.036 | 5.669 | 2237.265 |
| sx_imitation_quiet | L2 | 300 | 113 | 0.377 | 0.324 | 0.433 | 9.114 | 7.226 | 0.064 | 29.366 | 2128.541 |
| sx_imitation_quiet | L3 | 300 | 43 | 0.143 | 0.108 | 0.188 | 14.964 | 14.700 | 0.118 | 88.396 | 2071.047 |
| sx_retrained | L0 | 300 | 300 | 1.000 | 0.987 | 1.000 | 1.717 | 1.616 | 0.005 | 1.052 | 2248.749 |
| sx_retrained | L1 | 300 | 259 | 0.863 | 0.820 | 0.898 | 1.852 | 1.737 | 0.005 | 4.966 | 2081.331 |
| sx_retrained | L2 | 300 | 161 | 0.537 | 0.480 | 0.592 | 3.205 | 1.877 | 0.044 | 26.919 | 1958.393 |
| sx_retrained | L3 | 300 | 90 | 0.300 | 0.251 | 0.354 | 7.009 | 3.068 | 0.091 | 76.956 | 1885.190 |
| pair_safe_final | L0 | 600 | 599 | 0.998 | 0.991 | 1.000 | 2.034 | 1.979 | 0.004 | 1.156 | 2137.285 |
| pair_safe_final | L1 | 600 | 525 | 0.875 | 0.846 | 0.899 | 2.152 | 2.086 | 0.005 | 4.133 | 2025.544 |
| pair_safe_final | L2 | 600 | 353 | 0.588 | 0.549 | 0.627 | 2.973 | 2.205 | 0.036 | 21.084 | 1941.047 |
| pair_safe_final | L3 | 600 | 210 | 0.350 | 0.313 | 0.389 | 5.982 | 2.670 | 0.085 | 62.734 | 1885.518 |
| expert_tuned | L0 | 300 | 300 | 1.000 | 0.987 | 1.000 | 2.037 | 2.013 | 0.005 | 1.096 | 2260.431 |
| expert_tuned | L1 | 300 | 258 | 0.860 | 0.816 | 0.895 | 2.178 | 2.117 | 0.006 | 4.825 | 2098.084 |
| expert_tuned | L2 | 300 | 161 | 0.537 | 0.480 | 0.592 | 3.418 | 2.271 | 0.050 | 26.452 | 1975.902 |
| expert_tuned | L3 | 300 | 95 | 0.317 | 0.267 | 0.371 | 7.137 | 3.047 | 0.095 | 75.828 | 1901.021 |

## Failure reasons (counts)

| controller | level | none | too_fast | tilted | off_pad | out_of_fuel | crashed_other |
|---|---|---|---|---|---|---|---|
| sx_imitation_quiet | L0 | 298 | 2 | 0 | 0 | 0 | 0 |
| sx_imitation_quiet | L1 | 204 | 96 | 0 | 0 | 0 | 0 |
| sx_imitation_quiet | L2 | 113 | 186 | 0 | 0 | 0 | 1 |
| sx_imitation_quiet | L3 | 43 | 255 | 0 | 0 | 0 | 2 |
| sx_retrained | L0 | 300 | 0 | 0 | 0 | 0 | 0 |
| sx_retrained | L1 | 259 | 9 | 0 | 32 | 0 | 0 |
| sx_retrained | L2 | 161 | 75 | 0 | 64 | 0 | 0 |
| sx_retrained | L3 | 90 | 153 | 0 | 57 | 0 | 0 |
| pair_safe_final | L0 | 599 | 1 | 0 | 0 | 0 | 0 |
| pair_safe_final | L1 | 525 | 9 | 0 | 66 | 0 | 0 |
| pair_safe_final | L2 | 353 | 105 | 0 | 142 | 0 | 0 |
| pair_safe_final | L3 | 210 | 270 | 0 | 120 | 0 | 0 |
| expert_tuned | L0 | 300 | 0 | 0 | 0 | 0 | 0 |
| expert_tuned | L1 | 258 | 2 | 0 | 40 | 0 | 0 |
| expert_tuned | L2 | 161 | 68 | 0 | 71 | 0 | 0 |
| expert_tuned | L3 | 95 | 151 | 0 | 54 | 0 | 0 |
