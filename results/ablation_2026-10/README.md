# Ablation jobs for the GENIUS Olympiad paper (2026-10)

Nothing in the simulator, the success rule or the existing checkpoint was changed. Test seeds 0-299 were used only to test; nothing was tuned on them.
The "trained controller" is the existing checkpoint `models/checkpoints/rt1_final.zip` (SHA-256 `d8bc33ec2c4260080949f1d55f4077b7d40f812eb2c7bfba2bac7cba0fc98e86`; the paper's `sx_retrained`).

| file | content |
|---|---|
| `results.json` | all four jobs (plus the supplementary noise sweep), the statistics and the list of differences over 3 points |
| `job1_expert_baseline.md` | job 1 table |
| `job2_failure_modes.md` | job 2 tables |
| `job3_sensor_noise.md`, `job3_sensor_noise_fine.md` | job 3 tables (requested levels; supplementary smaller levels) |
| `job4_training_seeds.md` | job 4 tables |
| `flags.md` | every difference over 3 percentage points |
| `data/` | per-episode CSVs, observation ranges, training logs of the five seeds (`data/job4/logs`) |
| `scripts/` | the scripts |

## Commands run (project root `C:\Users\firdo\rocketeer\SpaceXperiment`)

```bash
# jobs 1 and 2: expert and trained controller, seeds 0-299, L0-L3 (about 150 s)
python paper_assets/scripts/run_eval.py --controllers expert_tuned,rt1_final --levels 0,1,2,3 --episodes 300 --seed0 0 --workers 12 --out results/ablation_2026-10/data/job1_job2_episodes.csv
python results/ablation_2026-10/scripts/analyze_job12.py

# job 3: sensor noise 0, 1, 2, 5 % (about 3 min), and the supplementary sweep 0.05, 0.1, 0.25, 0.5 %
python results/ablation_2026-10/scripts/noise_eval.py --workers 12
python results/ablation_2026-10/scripts/noise_fine.py --workers 12

# job 4: five training seeds one after the other, each followed by one test on seeds 0-299 (about 4 hours in total on this PC)
bash results/ablation_2026-10/scripts/run_job4.sh        # runs train_seed.py --seed 1 ... 5
python results/ablation_2026-10/scripts/make_results.py  # job 4 table, results.json, flags.md
```

A few lines of throw-away Python (not saved) probed the noise break point on 20 seeds at L0 with 0.05 to 1 % noise before the supplementary sweep was written; its numbers are not used.

## Results in short

**Job 1, expert baseline (300 episodes per level, 95 % Wilson interval):** L0 300/300 = 1.000 [0.987, 1.000]; L1 258/300 = 0.860 [0.816, 0.895]; L2 161/300 = 0.537 [0.480, 0.592]; L3 95/300 = 0.317 [0.267, 0.371]. L0 was not missing from the project: an earlier run (`paper_assets/before/episodes.csv`) already had 300/300; this run reproduces all four levels exactly.

**Job 2, failure modes** (the label is the first broken rule in the order out of fuel, too fast, tilt, off pad, other; the types add up to the failures in all 8 rows):

| controller | level | failures | too fast | off pad | tilt too large | out of fuel | other |
|---|---|---|---|---|---|---|---|
| expert | L1 | 42 | 2 | 40 | 0 | 0 | 0 |
| expert | L2 | 139 | 68 | 71 | 0 | 0 | 0 |
| expert | L3 | 205 | 151 | 54 | 0 | 0 | 0 |
| trained | L1 | 41 | 9 | 32 | 0 | 0 | 0 |
| trained | L2 | 139 | 75 | 64 | 0 | 0 | 0 |
| trained | L3 | 210 | 153 | 57 | 0 | 0 | 0 |

L0 has no failures for either controller. The trained controller at L3 reproduces your check: 90 successes, 210 failures, 153 too fast, 57 off pad. `tilt too large` is always 0 only because of the label order; counted as "every rule broken" (second table in `job2_failure_modes.md`) tilt is involved in 49 of the 139 L2 failures and 102 of the 210 L3 failures of the trained controller, and off pad is involved in every failure at L1-L3.

**Job 3, sensor noise (trained controller, not retrained):** 0 % reproduces 300 / 259 / 161 / 90. At 1 %, 2 % and 5 % the success is **0/300 at every level** (Wilson upper bound 0.013). The supplementary sweep shows where it breaks: 0.05 % no change (L1 257, L2 160, L3 90); 0.1 % L1 249 (-3.3 points), L0 292, L2 157, L3 90; 0.25 % L0 182, L1 154, L2 110, L3 57; 0.5 % L0 29, L1 27, L2 20, L3 6.

**Job 4, five training seeds (1-5), each tested once on seeds 0-299:**

| level | mean | SD | min | max | current checkpoint |
|---|---|---|---|---|---|
| L0 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| L1 | 0.862 | 0.006 | 0.853 | 0.867 | 0.863 |
| L2 | 0.535 | 0.013 | 0.520 | 0.553 | 0.537 |
| L3 | 0.311 | 0.003 | 0.307 | 0.313 | 0.300 |

Per-seed counts are in `job4_training_seeds.md`.

## Flags (differences from the current numbers over 3 percentage points)

- Job 1: none (expert 0.860 / 0.537 / 0.317 against the quoted 0.86 / 0.54 / 0.32).
- Job 2: none (all numbers match the paper's).
- Job 4: **none**; no seed differs from the current checkpoint by more than 2 points at any level (the largest is +1.7 at L2, seed 1, which is 5 more successes out of 300).
- Job 3: every noise cell from 0.25 % upward, and L1 at 0.1 % (-3.3 points), differs by more than 3 points from the clean run; the full list is in `flags.md`.

## Things you should know before using these numbers

1. **"Range of each observation" is my definition.** The environment declares every observation in [-20, 20], which would make 1 % a noise of 0.4 on observations that are scaled to about [-1, 1]. I used each observation's observed span (maximum minus minimum) over clean flights of the same controller on the validation seeds 100000-100199 (`data/obs_ranges.json`), and noise = % of that span, independent at each 20 Hz step. With these spans, 1 % is about 30 m on the altitude reading, 52 m on the horizontal position, 4.6 m/s on the horizontal speed and 3 m/s on the vertical speed, against a success limit of 3 m/s at touchdown. So "1 % of the range" is a large error for a landing, which is why the controller collapses. If you meant a smaller or differently defined range, the supplementary sweep (0.05-0.5 %) is the nearest answer; I can rerun with another definition.
2. **The seed SD is small because the seeds share a start.** The existing recipe starts every run from the same checkpoint (`sx_imitation_quiet`) and tests on the same 300 episodes. A training seed changes the demonstration and DAgger episodes (their environment seeds are offset by seed x 10,000,000, so no two seeds share an episode and none overlaps the test or validation seeds), the level draws and the shuffling in the fit. The SD (0.3-1.3 points) is therefore far below the sampling uncertainty of a 300-episode test (Wilson half-width about 5 points) and does not include variation from the starting network or from the test sample. The current checkpoint's L3 (0.300) is slightly below all five new seeds (0.307-0.313); that is a 1-point difference, within noise.
3. **Training took longer than the earlier retrain** (30-53 minutes per seed against 17 minutes) on the same code. The recipe is identical; I do not know the cause.
4. Two launches of job 4 were started by mistake at the same time and were stopped and deleted before any result was produced (a missing `if __name__ == "__main__"` guard and a leftover process). The reported seeds come from the third, clean launch. No result from the aborted launches is used.
5. The training seeds are 1-5; the current checkpoint is the seed-0 run of the same recipe and is not included in the mean and SD.
