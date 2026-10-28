# Scripted expert: why it fails (test seeds 0-299, 300 episodes per level, noise 0)

## 1. Failure reasons per level (one label per failed episode: first match of out_of_fuel > too_fast > tilted > off_pad > crashed_other)

| level | episodes | failures | too_fast | tilted | off_pad | out_of_fuel | crashed_other |
|---|---|---|---|---|---|---|---|
| L0 | 300 | 0 | 0 | 0 | 0 | 0 | 0 |
| L1 | 300 | 42 | 2 | 0 | 40 | 0 | 0 |
| L2 | 300 | 139 | 68 | 0 | 71 | 0 | 0 |
| L3 | 300 | 205 | 151 | 0 | 54 | 0 | 0 |

## 2. Every rule that was broken (an episode can break several; counts of FAILED episodes that broke each rule)

| level | failures | off_pad | too_fast | tilted | not_at_rest | out_of_fuel | off_pad AND too_fast |
|---|---|---|---|---|---|---|---|
| L0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| L1 | 42 | 42 | 2 | 1 | 4 | 0 | 2 |
| L2 | 139 | 139 | 68 | 57 | 71 | 0 | 68 |
| L3 | 205 | 205 | 151 | 107 | 141 | 0 | 151 |

## 3. Size of the failures

| level | outcome | n | median pad offset (m) | median touchdown speed (m/s) | median fuel left (kg) |
|---|---|---|---|---|---|
| L1 | success | 258 | 1.520 | 2.075 | 2100.567 |
| L1 | failed | 42 | 18.007 | 2.675 | 2108.849 |
| L2 | success | 161 | 2.359 | 2.068 | 1949.910 |
| L2 | failed | 139 | 39.230 | 2.960 | 2014.277 |
| L3 | success | 95 | 3.675 | 2.072 | 1854.195 |
| L3 | failed | 205 | 83.257 | 7.883 | 1934.232 |

## 4. Success rate by how far the start is from the pad, and by wind (L1-L3 pooled per level; terciles of the quantity within each level)

| level | quantity | lowest third | middle third | highest third |
|---|---|---|---|---|
| L1 | |start x offset| (cuts 512.5 / 720.9) | 0.91 (n=100) | 0.90 (n=100) | 0.77 (n=100) |
| L1 | |wind mean| (cuts 1.0 / 2.0) | 0.80 (n=100) | 0.87 (n=100) | 0.91 (n=100) |
| L1 | |start vx| (cuts 43.1 / 59.2) | 0.91 (n=100) | 0.86 (n=100) | 0.81 (n=100) |
| L1 | start vy (fall speed) (cuts 155.7 / 168.3) | 0.83 (n=100) | 0.84 (n=100) | 0.91 (n=100) |
| L2 | |start x offset| (cuts 827.0 / 1158.3) | 0.67 (n=100) | 0.54 (n=100) | 0.40 (n=100) |
| L2 | |wind mean| (cuts 1.9 / 4.1) | 0.45 (n=100) | 0.63 (n=100) | 0.53 (n=100) |
| L2 | |start vx| (cuts 79.6 / 106.2) | 0.62 (n=100) | 0.51 (n=100) | 0.48 (n=100) |
| L2 | start vy (fall speed) (cuts 196.7 / 212.0) | 0.47 (n=100) | 0.53 (n=100) | 0.61 (n=100) |
| L3 | |start x offset| (cuts 1232.7 / 1691.9) | 0.43 (n=100) | 0.29 (n=100) | 0.23 (n=100) |
| L3 | |wind mean| (cuts 2.9 / 6.1) | 0.26 (n=100) | 0.40 (n=100) | 0.29 (n=100) |
| L3 | |start vx| (cuts 132.1 / 168.8) | 0.37 (n=100) | 0.28 (n=100) | 0.30 (n=100) |
| L3 | start vy (fall speed) (cuts 225.5 / 247.0) | 0.29 (n=100) | 0.30 (n=100) | 0.36 (n=100) |

## 5. Direction of the miss

- L1: of 42 failures that touched down, the landing point is on the same side of the pad as the wind push in 0.52 and on the same side as the start offset in 0.83; median |final x| 18.0 m.
- L2: of 139 failures that touched down, the landing point is on the same side of the pad as the wind push in 0.55 and on the same side as the start offset in 0.81; median |final x| 39.2 m.
- L3: of 205 failures that touched down, the landing point is on the same side of the pad as the wind push in 0.55 and on the same side as the start offset in 0.76; median |final x| 83.3 m.
