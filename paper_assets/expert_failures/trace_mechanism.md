# Is the expert's miss decided before the burn?  (test seeds 0-299, expert_tuned, noise 0; flown again with a per-step log, nothing in the expert or simulator changed)

Ignition state = height, sideways position x and sideways speed vx at the first control step with the engine latched on. `zem` = x + vx * t_go with t_go = 2h/(fall speed + V_TOUCH): where the stage would touch down relative to the pad if nothing were done sideways from then on. `demand` = the constant sideways acceleration that would bring x and vx to zero together at t_go, divided by the sideways acceleration the thrust could give with the tilt at its cap (average over the burn above 30 m); above 1 the tilt cap alone makes the miss unavoidable. `sat` = the tilt the expert asks for is at its cap (min(LATERAL_TILT_MAX, GROUND + PER_M*h)).

Check against the paper: successes L1 258/300, L2 161/300, L3 95/300 (paper: expert_tuned 258 / 161 / 95).

## 1. Ignition state of successes and failures (medians)

| level | outcome | n | ignition height (m) | abs x at ignition (m) | abs vx (m/s) | abs zem (m) | share of burn with tilt at cap | cap hit below 30 m |
|---|---|---|---|---|---|---|---|---|
| L1 | success | 258 | 970 | 136.6 | 31.4 | 198.9 | 0.07 | 1.00 |
| L1 | failed | 42 | 965 | 251.0 | 36.0 | 127.1 | 0.48 | 1.00 |
| L2 | success | 161 | 1099 | 245.6 | 55.4 | 364.3 | 0.18 | 1.00 |
| L2 | failed | 139 | 1067 | 381.6 | 59.0 | 245.5 | 0.60 | 0.99 |
| L3 | success | 95 | 1171 | 382.1 | 88.8 | 557.2 | 0.29 | 1.00 |
| L3 | failed | 205 | 1141 | 540.5 | 91.4 | 397.4 | 0.71 | 0.98 |

## 2. How well does each quantity tell failures from successes?  (AUC: 0.5 = no information, 1.0 = every failure ranks above every success)

| level | demand / capacity | abs zem at ignition | abs x at ignition | abs vx at ignition | abs start x | abs wind mean | abs x at 100 m | abs vx at 100 m | share of burn at cap |
|---|---|---|---|---|---|---|---|---|---|
| L1 | 0.29 | 0.31 | 0.81 | 0.61 | 0.69 | 0.37 | 1.00 | 0.97 | 1.00 |
| L2 | 0.25 | 0.29 | 0.78 | 0.56 | 0.65 | 0.46 | 1.00 | 0.96 | 1.00 |
| L3 | 0.28 | 0.29 | 0.77 | 0.57 | 0.64 | 0.48 | 0.99 | 0.83 | 0.98 |

## 3. Success rate by demand / capacity at ignition

| demand / capacity | L1 | L2 | L3 |
|---|---|---|---|
| 0 to 0.25 | 0.75 (n=134) | 0.18 (n=95) | 0.00 (n=71) |
| 0.25 to 0.5 | 0.99 (n=109) | 0.58 (n=81) | 0.15 (n=59) |
| 0.5 to 0.75 | 0.92 (n=51) | 0.99 (n=71) | 0.51 (n=57) |
| 0.75 to 1 | 0.50 (n=6) | 0.67 (n=36) | 0.79 (n=57) |
| 1 to 1.5 | - | 0.18 (n=17) | 0.28 (n=43) |
| 1.5 to inf | - | - | 0.00 (n=13) |

Note: the zero-effort miss alone does not separate the groups (its AUC is below 0.5): failures often start with a large sideways speed already pointing at the pad, so x + vx*t_go is small, but stopping that speed is what costs tilt. The demand / capacity ratio includes both.

## 4. Does the steering run out of authority, and where?  (burn above 150 m, steps where the expert asks for more than 0.2 rad of tilt)

| level | outcome | tilt at cap for more than half of the burn | median height of first cap hit (m) | tilt delivered / tilt asked (median) | gimbal at its limit | fins at their limit |
|---|---|---|---|---|---|---|
| L1 | success | 0.00 | 951 | 0.22 | 0.00 | 0.00 |
| L1 | failed | 0.33 | 965 | 0.21 | 0.00 | 0.45 |
| L2 | success | 0.00 | 1070 | 0.22 | 0.00 | 0.00 |
| L2 | failed | 0.79 | 1067 | 0.13 | 0.00 | 0.60 |
| L3 | success | 0.01 | 1135 | 0.24 | 0.00 | 0.09 |
| L3 | failed | 0.87 | 1126 | 0.02 | 0.00 | 0.63 |

AUC of tilt delivered / tilt asked for telling failures from successes: L1 0.53, L2 0.70, L3 0.61 (lower delivered tilt in failures = higher AUC).

Episodes where the engine never latched on: 0 of 900.
