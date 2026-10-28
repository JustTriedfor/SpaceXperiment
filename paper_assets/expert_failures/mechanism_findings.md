# Why the expert misses at L2 and L3, and was its tuning held back?  (summary of two new runs; the expert and simulator were not changed)

Scripts: `trace_mechanism.py` (tables in `trace_mechanism.md`, per-episode rows in `trace_mechanism.csv`), `tune_wider.py` (results in `tune_wider.md` and `tune_original_limits.md`, logs `*.log`, best constants `*_best.json`). Both reproduce the paper's expert numbers first (258 / 161 / 95 successes at L1 / L2 / L3 on seeds 0-299).

## 1. Is the miss decided before the burn?  No.

- A simple "too much sideways error at ignition" does not explain failures. The sideways acceleration needed at ignition (relative to what the tilt cap allows) is not higher in failures; at L3 the success rate is 0.00 for the lowest-demand group and 0.79 for the group just below 1.0 (`trace_mechanism.md`, table 3). The zero-effort miss alone is even lower in failures (AUC about 0.3), because failures often start with a large sideways speed that already points at the pad.
- The failures do show a different burn. In failures the tilt the expert asks for sits at its cap for 79 % (L2) and 87 % (L3) of the burn (median over episodes, more than half of the burn), in successes for 0-1 %. Fins are at their limit during 45-63 % of the burn above 150 m in failures and 0-9 % in successes; the gimbal is never at its limit.
- Above 150 m the stage delivers only a small part of the tilt that is asked for: the median delivered / asked ratio is 0.22 (L1 successes) down to 0.02 (L3 failures). Three printed L3 flights (seeds 0, 5, 6; not in the repo) show the same: th_des is +-40 deg while the actual tilt stays within about 10 deg until the last 100-150 m.

What this supports (association, not proof): the expert's sideways steering gets almost no effect while the stage is fast and high, and starts working only in the last few hundred metres; failures are the episodes where that late window is too short for the start offset and sideways speed. What it does not show: the physical reason the tilt is not delivered (airflow torque against the fins, the attitude loop's bandwidth `ATT_WN_MAX`, or the actuator split). That was not isolated, and no actuator-level experiment was run. The sentence for the paper that the data support is: "most failures end more than 10 m from the pad; the expert cannot turn the stage far enough early in the descent to bring it over the pad", not "it fails by touching down too fast".

## 2. Was the tuning held back by the search limits?  Not by the two edge limits, but the tuning was not finished.

Same random hill-climb as `tools/expert_tuning/tune_expert.py` (same objective, tuner seeds 10,000+, 300 iterations), starting from the expert as it is now:

| run | L1 | L2 | L3 | fresh-seed soft-landing rate L1 / L2 |
|---|---|---|---|---|
| current expert (paper) | 258 | 161 | 95 | 0.773 / 0.520 |
| 300 more iterations, original limits (control) | 266 | 178 | 122 | 0.810 / 0.610 |
| 300 more iterations, LATERAL_TILT_MAX up to 1.2 and ALIGN_RAMP_M up to 6000 | 267 | 180 | 121 | 0.813 / 0.595 |

(successes out of 300 on the paper's test seeds 0-299, looked at once after the search; the constants were chosen on other seeds). Widening the two limits made no difference (the best TILT_MAX found was 0.757 and ALIGN_RAMP_M went down). The gain comes from continuing the search: the published expert is not at the optimum of its own tuner. LATERAL_TILT_PER_M now sits on its own edge (0.012) in both runs, so another constant is limiting, not the two I widened. L0 stays 300 / 300.

Caveats: one random-search seed per run, so differences of a few episodes between the two new runs are noise; the tuner objective counts L0-L2 only (L3 improved anyway); the tuner's crash rate at L2 rose slightly (0.165 to 0.18) while the soft-landing rate rose.

## What this means for the paper (decision for the author, nothing was changed)

- The paper's expert and the retrained controller (which imitates it) have L2/L3 success rates about 17 and 27 episodes (of 300) below what the same expert code reaches with 300 more tuning iterations. If that matters for the story, the options are: keep the numbers and say plainly that the tuning was stopped early, or re-tune, re-run `paper_assets/scripts/run_eval.py` and retrain, which changes every downstream number, table and figure (Figures 5 and 7 included).
- Not done in this round: the `COAST_STEER` and other switched-off options, sensor noise on the expert, and the fuel-budget check.
