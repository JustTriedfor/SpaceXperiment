# Tuning again with the two edge limits widened (new run; the expert in src/ is unchanged)

Iterations: 300; same objective, seeds and steps as `tools/expert_tuning/tune_expert.py`; start = the expert as it is now. Widened: LATERAL_TILT_MAX up to 1.2 rad (was 0.7), ALIGN_RAMP_M up to 6000 m (was 2500).

| constant | start (current expert) | best found |
|---|---|---|
| A_PLAN_FRACTION | 0.6364 | 0.53981 |
| LATERAL_TILT_MAX | 0.7 | 0.75657 |
| LATERAL_TILT_GROUND | 0.16518 | 0.18692 |
| LATERAL_TILT_PER_M | 0.0095481 | 0.012 |
| LATERAL_VX_GAIN | 3.6486 | 3.6486 |
| LATERAL_TGO_SCALE | 0.38003 | 0.57734 |
| LATERAL_TGO_MIN | 3.4092 | 3.4092 |
| LATERAL_X_FADE_M | 269.94 | 238.46 |
| LATERAL_LEAD_S | 1.2756 | 1.2831 |
| ALIGN_RAMP_M | 2500 | 2347.3 |
| ATT_WN_MAX | 0.79443 | 0.77766 |
| V_TOUCH | 1.775 | 1.2794 |

## Tuner objective (lower is better) and results on the tuner's own seeds 10,000+

start 0.4707, best 0.3867.

## Fresh seeds 500,000+ (the tuner's validation set): crash rate L0/L1/L2 and soft-landing rate

| | crash L0/L1/L2 | soft L0/L1/L2 |
|---|---|---|
| start | 0.000/0.000/0.165 | 1.000/0.773/0.520 |
| best | 0.000/0.000/0.175 | 1.000/0.813/0.595 |

## Paper test seeds 0-299 (looked at once, after the search; paper success rule), successes of 300 and median miss of the failures

| | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| start | 300 (miss 0.0 m) | 258 (miss 18.0 m) | 161 (miss 39.2 m) | 95 (miss 83.3 m) |
| best | 300 (miss 0.0 m) | 267 (miss 18.2 m) | 180 (miss 42.3 m) | 121 (miss 94.5 m) |
