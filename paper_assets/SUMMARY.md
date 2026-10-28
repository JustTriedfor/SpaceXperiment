# Summary of all paper data, tables and files (GENIUS Olympiad / REBOUND)

Project root: `C:\Users\firdo\rocketeer\SpaceXperiment`. Paths below are relative to it. The project is not under git; code and checkpoints are identified by SHA-256 (listed where relevant). Test seeds are 0-299 at every level; validation seeds 100000-100199; training seeds are never in those ranges.

## 1. Main evaluation (`paper_assets/`, run on 2026-10-21; old numbers kept unchanged in `paper_assets/before/`)

Success = touchdown speed <= 3 m/s, tilt <= 0.15 rad, within 10 m of the pad, never out of propellant, ended at rest. Noise 0 (deterministic). 3,600 episodes (3 controllers x L0-L3 x 300 seeds). Wind scale 0 / 5 / 10 / 15 m/s is a scale, not a constant: each episode draws a mean within +-0.6 x scale plus gusts of 0.4 x scale (`paper_assets/wind_levels.txt`).

| controller (successes / n, rate [95 % Wilson]) | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| before: `sx_imitation_quiet` (n = 300) | 298 = 0.993 [0.976, 0.998] | 204 = 0.680 [0.625, 0.730] | 113 = 0.377 [0.324, 0.433] | 43 = 0.143 [0.108, 0.188] |
| after retraining: `rt1_final` (n = 300) | 300 = 1.000 [0.987, 1.000] | 259 = 0.863 [0.820, 0.898] | 161 = 0.537 [0.480, 0.592] | 90 = 0.300 [0.251, 0.354] |
| scripted expert (n = 300) | 300 = 1.000 [0.987, 1.000] | 258 = 0.860 [0.816, 0.895] | 161 = 0.537 [0.480, 0.592] | 95 = 0.317 [0.267, 0.371] |
| pair `pair_safe_final`, per lander (n = 600) | 599 = 0.998 [0.991, 1.000] | 525 = 0.875 [0.846, 0.899] | 353 = 0.588 [0.549, 0.627] | 210 = 0.350 [0.313, 0.389] |
| pair, joint (n = 300 episodes) | 299 = 0.997 | 258 = 0.860 | 172 = 0.573 | 96 = 0.320 |

Clean trials needed for a one-sided 95 % bound: p >= 0.99 needs 299 (exact binomial) or 300 (rule of three); p >= 0.9963 needs 809 or 811. Only the expert at L0 (300/300, bound 0.990) reaches 0.99; nothing reaches 0.9963.
Failure reasons of the old single controller: almost all `too_fast` (L3: 255 of 257); expert and pair also fail `off_pad`. No `out_of_fuel` and no `tilted` label anywhere (tilt is hidden by the label order).

| file | content |
|---|---|
| `paper_assets/episodes.csv` | every episode (4,800 rows): controller, level, wind_mean_mps, seed, touchdown_speed_mps, tilt_rad, pad_offset_m, fuel_remaining_kg, success, failure_reason (+ extra columns) |
| `paper_assets/summary.csv`, `summary.md` | per controller and level: n, successes, rate, Wilson interval, mean / median speed, mean tilt, mean offset, mean fuel, failure counts, rule-of-three table |
| `paper_assets/success_rules.txt`, `wind_levels.txt` | the four success rules with code and thresholds; wind definition and realised wind statistics |
| `paper_assets/chart_success_by_level.png`, `chart_touchdown_speed_hist.png` | the two charts (white, Times-style) |
| `paper_assets/fig3_1_interface_full.png`, `fig3_2_recording_panels.png`, `fig4_1_landing_scatter.png`, `fig4_2_descent_phase_portrait.png`, `fig4_2b_decision_boundary.png` | UI captures (1920x1080 at 2x) of one landing: `sx_imitation_quiet`, level 1, **seed 122** (2.416 m/s, 2.379 m offset). The scatter holds the 1,200 `sx_imitation_quiet` touchdowns; the decision-boundary footnote is cropped because it overlaps in the viewer |
| `paper_assets/README.md`, `run_eval.log`, `run_eval.time` | commands, hashes, runtime (160 s) |
| `paper_assets/scripts/` | `run_eval.py` (evaluator, extended with `--seed0`), `make_rules.py`, `make_summary.py`, `make_charts.py`, `make_live_episode.py`, `capture_ui.py` |
| `paper_assets/before/` | read-only copy of the old results (`episodes.csv`, `summary.*`, charts, `run_eval.py.orig`) |

Checkpoints (`models/checkpoints/`): `sx_imitation_quiet.zip` (SHA-256 `a71c329b...324e7f`), `pair_safe_final_A.zip` (`fde913c8...474179`), `pair_safe_final_B.zip` (`589138cb...ab0a1`), retrained `rt1_final.zip` = `rt1_dagger_4` (`d8bc33ec2c4260080949f1d55f4077b7d40f812eb2c7bfba2bac7cba0fc98e86`). Expert: `src/sim/expert.py` (`bc17fccb...e569b4`).

## 2. Retraining the single-lander controller (`paper_assets/retrain/`)

Why the pair was better: the single controller was cloned against the expert *before* tuning, with 1,360 episodes (604,648 steps) on levels 0,0,1,1,2; the pair used the tuned expert, 1,700 episodes (976,071 steps), more level-1 wind. Recipe of the one attempt (no second one was needed): start from `sx_imitation_quiet`, 800 expert demonstrations (734 kept) on levels 0,1,1,2,2,3, cloning 40 epochs, 8 DAgger rounds of 180 episodes on levels 0,1,1,2,2,3,3, 20 epochs per round (batch 1024, lr 5e-4, 1e-4 in the last two rounds); final round chosen on the validation seeds (round 4, mean 0.638).

| validation (200 per level) | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| expert | 1.000 | 0.840 | 0.465 | 0.235 |
| old `sx_imitation_quiet` | 0.990 | 0.655 | 0.360 | 0.145 |
| `rt1_dagger_4` (chosen) | 1.000 | 0.840 | 0.485 | 0.225 |

Test result: see the table in section 1 (within 1.7 points of the expert at every level; mean speed L1 4.268 to 1.852 m/s, L3 14.964 to 7.009 m/s). Training took 1,015 s.

Files: `paper_assets/retrain/README.md` (commands, seeds, why this checkpoint, caveats), `attempts.md` (every round's validation scores and two aborted infrastructure launches), `episodes.csv`, `summary.csv`, `summary.md`, `test_new.csv`, `chart_success_by_level_after.png`, `chart_touchdown_speed_hist_after.png`, `frozen_hashes_start.txt`, `frozen_hashes_end.txt` (only `run_eval.py` changed), `attempt1.json`, `scripts/retrain_single.py`, `fit_worker.py`, `make_after.py`, `val/`, `logs/`. Training data: `data/trajectories/rt1_*`, `logs/rt1_*`; checkpoints `models/checkpoints/rt1_*`.

## 3. Why the expert fails at L2/L3 (`paper_assets/expert_failures/`)

| expert, 300 episodes | failures | too fast | off pad | tilted | out of fuel |
|---|---|---|---|---|---|
| L1 | 42 | 2 | 40 | 0 | 0 |
| L2 | 139 | 68 | 71 | 0 | 0 |
| L3 | 205 | 151 | 54 | 0 | 0 |

Every failure at L1-L3 broke the 10 m pad rule (median miss 18.0 / 39.2 / 83.3 m against 1.5 / 2.4 / 3.7 m for successes); fuel is never the limit; misses fall on the side of the start offset (76-83 %), not of the wind (52-55 %); success falls with the start offset (L2: 0.67 / 0.54 / 0.40 by thirds; L3: 0.43 / 0.29 / 0.23). The levels also raise start offset, speed and entry angle, so L3 is not only windier. Association only, not a tested mechanism. Files: `README.md`, `failure_table.md`, `failure_by_start.csv`, `analyze.py`.

## 4. Development history (`paper_assets/progress/`)

No git history exists, so dates are file-system times, there are no commit hashes and old versions cannot be run. Two stages: the Rocketeer prototype (gymnasium Box2D lunar lander, 18 Oct 2026) and SpaceXperiment with own 2-D physics (from 19 Oct 2026 21:06).

| item | value |
|---|---|
| first file / last source change | 2026-10-18 12:57 (`.gitignore`; first source 13:14) / 2026-10-21 07:25 |
| commits | none |
| prototype code | 31 Python files / 5,448 lines; 1 HTML-JS-CSS page / 1,914 lines; 2 Markdown / 430 lines |
| SpaceXperiment code | 57 Python files / 8,347 lines; 1 page / 3,311 lines; 4 Markdown / 917 lines |
| tests | prototype 168 passed (60 s); SpaceXperiment 130 passed (119 s) |
| milestones | 15 (`milestones.md`), e.g. levels L0-L3 with wind 18 Oct 14:34, own physics 19 Oct 21:06, scripted expert 20 Oct 05:27, dual landing 20 Oct 09:13 |

Training logs plotted unsmoothed (`train_custom_s0_m2.png`, `train_custom_s0_staged8.png`, `train_custom_s0_quiet.png`, `train_custom_s0_pair.png`; sources in `training_runs.md`): PPO from scratch never reached strict success; PPO fine-tuning of the imitation network made it worse (`custom_s0_quiet`: soft-landing share 0.33 to about 0).
Images: `img_01_2026-10-18_lunarlander-prototype-gif-frame.png` (frame 43 of the existing GIF), `img_02_2026-10-19_rocketeer-box2d-viewer.png` (prototype viewer, run `custom_s0_staged8` episode 4428), `img_03_2026-10-21_spacexperiment-current.png` (seed 122); composite `fig3_1_early_versions.png`; `timeline.png`. Existing recordings copied unedited to `paper_assets/progress/existing/` (`custom_l0_m2.gif`, `custom_l0_locked.gif`, `custom_l0_m2_progress.png`). Also `README.md` (with a 103-word paragraph for the paper), `file_history.txt`, `milestones.md`, `scripts/`. Not reproduced: intermediate versions (no snapshots).

## 5. Emissions Monte Carlo (`paper_assets/montecarlo/`)

The original script was not found; it was rebuilt from Chapter 3 (200,000 draws, triangular inputs from Table 3.1, N uniform 15-40, seed 20260101). The formulas are inferred: E = (1 - p^N)/(1 - p); C = P1(1 + r(E - 1))/E + P2 + F + T; expendable = P1 + P2 + F + T (mean 998.55 t). They reproduce Table 4.1 within 1 t and 1 point (differences up to +0.60 t at p = 0.9963):

| p | expected flights | mean t CO2e / launch | 5th - 95th pct | saving vs expendable |
|---|---|---|---|---|
| 0.33 | 1.49 | 868.45 | 815.7 - 920.4 | 13.03 % |
| 0.60 | 2.50 | 762.02 | 719.4 - 805.1 | 23.69 % |
| 0.90 | 9.26 | 647.06 | 610.7 - 684.7 | 35.20 % |
| 0.95 | 14.76 | 631.58 | 595.3 - 669.2 | 36.75 % |
| 0.99 | 23.95 | 621.87 | 585.5 - 659.5 | 37.72 % |
| 0.9963 | 26.12 | 620.60 | 584.2 - 658.2 | 37.85 % |
| **L3, p = 0.300** | 1.43 | 880.3 | 826.2 - 933.5 | 11.84 % (118.3 t) |
| **L2, p = 0.537** | 2.16 | 786.9 | 742.2 - 831.7 | 21.20 % (211.7 t) |
| **L1, p = 0.863** | 7.08 | 660.1 | 623.5 - 698.0 | 33.90 % (338.5 t) |
| **L0, p = 0.99** (lower bound of 300/300) | 23.95 | 621.9 | 585.5 - 659.5 | 37.72 % (376.7 t) |

Effect of the finite sample (Wilson interval of p): saving L1 32.29 to 35.13 % (2.84 points), L2 18.95 to 23.37 % (4.42), L3 9.91 to 13.98 % (4.07). Files: `README.md`, `results.csv` (full precision), `results.md`, `results_check_continuous_n.csv`, `emissions_mc.py` (SHA-256 `6bd18160...a6aa7a9c`), `make_outputs.py`, `fig4_3_emissions_vs_p.png` (1600x900).

## 6. Proposal figures (`paper_assets/proposal/`)

Made by `make_proposal_figs.py --lang en|id`: `fig2_1_expected_flights_{en,id}.png/.svg` (5.5 x 2.4 in, 1650x720 px), `fig3_3a_chain_{en,id}.png/.svg` and `fig3_3b_training_{en,id}.png/.svg` (2.6 x 2.9 in, 780x870 px), `fig2_1_values.csv` (p, E for N = 15, 25, 40 on a 0.005 grid).

## 7. Ablation jobs (`results/ablation_2026-10/`, run 2026-10-26)

Master file `results/ablation_2026-10/results.json`; tables `job1_expert_baseline.md`, `job2_failure_modes.md`, `job3_sensor_noise.md`, `job3_sensor_noise_fine.md`, `job4_training_seeds.md`; `flags.md`; `README.md` (every command); `data/` (episodes, obs ranges, `data/job4/` logs and tests); `scripts/`.
- Job 1 (expert) and Job 2 (failure modes): same numbers as sections 1 and 3 (trained L3: 90 / 210 / 153 too fast / 57 off pad, matches).
- Job 3, noise (trained controller, noise = % of each observation's measured span): 0 % = 300 / 259 / 161 / 90; **1 %, 2 % and 5 % = 0 / 300 at every level**. Smaller levels: 0.05 % L1 257, L2 160, L3 90; 0.1 % L0 292, L1 249, L2 157, L3 90; 0.25 % L0 182, L1 154, L2 110, L3 57; 0.5 % L0 29, L1 27, L2 20, L3 6. 1 % is about 30 m on altitude and 3 m/s on vertical speed.
- Job 4, five training seeds (successes of 300 at L0-L3): seed 1 = 300/259/166/94; seed 2 = 300/256/156/94; seed 3 = 300/258/159/92; seed 4 = 300/260/159/93; seed 5 = 300/260/163/94. Mean (SD) rate: L0 1.000 (0.000), L1 0.862 (0.006), L2 0.535 (0.013), L3 0.311 (0.003); current checkpoint 1.000 / 0.863 / 0.537 / 0.300. No difference over 3 points. The SD is small because the seeds share the starting checkpoint and the test seeds.

## Things to keep in mind when citing

- Dates in the development history are file times, not commits.
- The emissions model is a rebuild with inferred formulas; the paper's Section 2.5 was not available.
- Pair per-lander rows are not independent (shared wind): use the joint rows for the pair.
- The retrained controller imitates the expert, so its L2/L3 ceiling is the expert's.
- Test seeds 0-299 were used only for the final tests; the retraining, its checkpoint choice and the noise ranges used validation seeds 100000-100199.
