# Job 2: failure modes (seeds 0-299)

One label per failed episode, first match of: out_of_fuel, too_fast (first-contact speed > 3 m/s), tilted (> 0.15 rad), off_pad (> 10 m), otherwise other (no touchdown, fall over, failed settle). The columns add up to the failures; the last column checks it.

| controller | level | successes | failures | too fast | off pad | tilt too large | out of fuel | other | sum of types | sum = failures |
|---|---|---|---|---|---|---|---|---|---|---|
| scripted expert | L0 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes |
| scripted expert | L1 | 258 | 42 | 2 | 40 | 0 | 0 | 0 | 42 | yes |
| scripted expert | L2 | 161 | 139 | 68 | 71 | 0 | 0 | 0 | 139 | yes |
| scripted expert | L3 | 95 | 205 | 151 | 54 | 0 | 0 | 0 | 205 | yes |
| trained controller | L0 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes |
| trained controller | L1 | 259 | 41 | 9 | 32 | 0 | 0 | 0 | 41 | yes |
| trained controller | L2 | 161 | 139 | 75 | 64 | 0 | 0 | 0 | 139 | yes |
| trained controller | L3 | 90 | 210 | 153 | 57 | 0 | 0 | 0 | 210 | yes |

Every rule an episode broke (an episode can break several, so these do not add up to the failures; this shows how often each rule is involved):

| controller | level | failures | broke too_fast | broke off_pad | broke tilt | broke out_of_fuel |
|---|---|---|---|---|---|---|
| scripted expert | L0 | 0 | 0 | 0 | 0 | 0 |
| scripted expert | L1 | 42 | 2 | 42 | 1 | 0 |
| scripted expert | L2 | 139 | 68 | 139 | 57 | 0 |
| scripted expert | L3 | 205 | 151 | 205 | 107 | 0 |
| trained controller | L0 | 0 | 0 | 0 | 0 | 0 |
| trained controller | L1 | 41 | 9 | 41 | 0 | 0 |
| trained controller | L2 | 139 | 75 | 139 | 49 | 0 |
| trained controller | L3 | 210 | 153 | 210 | 102 | 0 |

Check against the paper's numbers (trained controller, L3): 90 successes, 210 failures, 153 too fast, 57 off pad: matches 90 / 210 / 153 / 57.
