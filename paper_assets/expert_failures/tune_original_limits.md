# Control: tuning again with the ORIGINAL limits (new run; the expert in src/ is unchanged)

Iterations: 300; same objective, seeds and steps as `tools/expert_tuning/tune_expert.py`; start = the expert as it is now. Limits as in the original tuner.

| constant | start (current expert) | best found |
|---|---|---|
| A_PLAN_FRACTION | 0.6364 | 0.61271 |
| LATERAL_TILT_MAX | 0.7 | 0.7 |
| LATERAL_TILT_GROUND | 0.16518 | 0.1978 |
| LATERAL_TILT_PER_M | 0.0095481 | 0.012 |
| LATERAL_VX_GAIN | 3.6486 | 3.6486 |
| LATERAL_TGO_SCALE | 0.38003 | 0.57734 |
| LATERAL_TGO_MIN | 3.4092 | 3.4092 |
| LATERAL_X_FADE_M | 269.94 | 235.64 |
| LATERAL_LEAD_S | 1.2756 | 1.2831 |
| ALIGN_RAMP_M | 2500 | 2097.9 |
| ATT_WN_MAX | 0.79443 | 0.77766 |
| V_TOUCH | 1.775 | 1.0726 |

## Tuner objective (lower is better) and results on the tuner's own seeds 10,000+

start 0.4707, best 0.3867.

## Fresh seeds 500,000+ (the tuner's validation set): crash rate L0/L1/L2 and soft-landing rate

| | crash L0/L1/L2 | soft L0/L1/L2 |
|---|---|---|
| start | 0.000/0.000/0.165 | 1.000/0.773/0.520 |
| best | 0.000/0.000/0.180 | 1.000/0.810/0.610 |

## Paper test seeds 0-299 (looked at once, after the search; paper success rule), successes of 300 and median miss of the failures

| | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| start | 300 (miss 0.0 m) | 258 (miss 18.0 m) | 161 (miss 39.2 m) | 95 (miss 83.3 m) |
| best | 300 (miss 0.0 m) | 266 (miss 17.8 m) | 178 (miss 37.5 m) | 122 (miss 95.7 m) |
