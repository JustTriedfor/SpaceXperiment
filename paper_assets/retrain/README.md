# Retraining the single-lander controller (step 1)

**Result: the retrained controller reaches the scripted expert's level on the held-out test seeds.** One recipe attempt was needed (`attempts.md`). The old results are kept unchanged in `paper_assets/before/` (read-only copies).

## Before / after (test seeds 0-299, 300 episodes per level, noise 0, 95 % Wilson intervals)

| controller | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| before: `sx_imitation_quiet` | 0.993 [0.976, 0.998] | 0.680 [0.625, 0.730] | 0.377 [0.324, 0.433] | 0.143 [0.108, 0.188] |
| **after: retrained (`rt1_final`)** | **1.000 [0.987, 1.000]** | **0.863 [0.820, 0.898]** | **0.537 [0.480, 0.592]** | **0.300 [0.251, 0.354]** |
| scripted expert (unchanged) | 1.000 [0.987, 1.000] | 0.860 [0.816, 0.895] | 0.537 [0.480, 0.592] | 0.317 [0.267, 0.371] |
| pair controller, per lander (unchanged) | 0.998 [0.991, 1.000] | 0.875 [0.846, 0.899] | 0.588 [0.549, 0.627] | 0.350 [0.313, 0.389] |

The retrained controller is within 1.7 percentage points of the expert at every level (L1 +0.3, L2 0.0, L3 -1.7), and its intervals overlap the expert's. The pair numbers are per lander; their landers share wind, so the intervals are narrower than the evidence (joint episode success of the pair: 0.860 / 0.573 / 0.320 at L1 / L2 / L3, see `before/summary.md`). The mean touchdown speed fell at every level, e.g. L1 4.268 to 1.852 m/s and L3 14.964 to 7.009 m/s. Full tables with failure reasons: `summary.md`, `summary.csv` (full precision); per-episode rows: `episodes.csv`; charts: `chart_success_by_level_after.png`, `chart_touchdown_speed_hist_after.png`.

## Why the pair controller was better (found by comparing the recipes, 7 lines)

1. **Different teacher.** `sx_imitation_quiet` (pipeline `pipe1`, 20 Oct) was cloned and DAgger-labelled by the expert as it was *before* the tuning of 20 Oct 17:27-17:31 (`tools/expert_tuning/expert_before_tune.py`); `pair_safe_final` was trained against the tuned expert that is evaluated here. The single-lander network was never retrained after the tuning.
2. **More data.** Single: 400 demonstrations (180,773 steps) + 6 DAgger rounds of 160 episodes = 1,360 episodes / 604,648 steps. Pair, per network: 500 demonstrations (291,580 steps) + 8 rounds of 150 = 1,700 episodes / 976,071 steps (1.6 times more).
3. **Levels.** Single: demonstrations and DAgger on levels 0,0,1,1,2. Pair: 0,0,1,1,1,2 and 0,1,1,1,2 (more wind). Neither ever trained on L3.
4. **Same network, loss and optimiser:** 256x256 MLP, behaviour-cloning MSE plus value loss, weight 5 on states below 400 m, lr 1e-3 (clone) then 5e-4, batch 512, exploration std set to exp(-1.6). The pair networks see a 40-input view (both landers), the single one 18 inputs.
5. **Pair started from copies of `sx_imitation_quiet`** and then trained on the tuned expert's labels.
6. Both choose the best DAgger round on 30 episodes (seeds 1000-1029) at levels 0-2 (single) or 0-1 (pair).
7. These are differences in the records, not ablations: I did not test which of them mattered most.

## What was done (recipe of the chosen attempt)

Start from `sx_imitation_quiet`; 800 expert demonstrations from the current tuned expert (734 kept) on levels 0,1,1,2,2,3; cloning 40 epochs; 8 DAgger rounds of 180 episodes on levels 0,1,1,2,2,3,3 (more wind, including L3); 20 epochs per round at batch 1024 (the pair used 30 epochs at 512; the change keeps GPU time short); lr 5e-4, and 1e-4 in the last two rounds; weight 5 on the landing phase. Same network size (256x256). Details and per-round validation scores: `attempts.md`, `logs/rt1.log`, `logs/rt1_result.json`.

## Seeds (nothing overlaps; the script asserts it)

| set | seeds | used for |
|---|---|---|
| **test** | 0-299 (every level) | the final evaluation only, run once on the chosen checkpoint, with the same script as before (`paper_assets/scripts/run_eval.py`, same 300 episodes per level) |
| **validation** | 100000-100199 (every level) | choosing the checkpoint after each DAgger round and the stop rule |
| **training** | expert demonstrations 1,000,000-1,000,895; DAgger round r: 2,000,000 + 100,000 r + i | demonstrations and DAgger data |

The test seeds were not used for training, expert data, DAgger, the choice of the round, early stopping or the stop rule. Disclosure: the old pipelines drew their training seeds randomly from 31 bits and evaluated on seeds 1000-1029; the expert tuning used seeds 10,000+ (validation 500,000+). None of these touch 0-299 by design, but the old data were not logged with their seeds, so an overlap by chance (probability about 1e-3) cannot be ruled out for the *old* controller.

## Checkpoint chosen and why

`rt1_final` = `rt1_dagger_4` (SHA-256 `d8bc33ec2c4260080949f1d55f4077b7d40f812eb2c7bfba2bac7cba0fc98e86`), the DAgger round with the highest mean validation success over the four levels (0.638 vs 0.614-0.638 for rounds 2-8). Rounds 4-8 are within the validation noise of one another; the later rounds were not looked at on the test seeds, so no round was picked by its test score.

## Frozen code

`frozen_hashes_start.txt` and `frozen_hashes_end.txt` hold SHA-256 of `src/sim/*.py`, `src/utils/config.py`, `src/training/*.py`, the old checkpoints and the evaluation scripts. Only one file differs: `paper_assets/scripts/run_eval.py` got a `--seed0` option (default 0) and now treats any other controller name as a single-lander checkpoint. The old file is saved in `before/run_eval.py.orig`; re-running the old controller on level 1 gives identical rows (checked, 300 of 300). There is no git repository, so hashes are the record. The success rule, level definitions, physics and expert were not changed.

## Commands

```bash
python paper_assets/scripts/run_eval.py --controllers expert_tuned,sx_imitation_quiet --levels 0,1,2,3 --episodes 200 --seed0 100000 --workers 12 --out paper_assets/retrain/val/baselines_val.csv   # validation baselines
python paper_assets/retrain/scripts/retrain_single.py --name rt1 --config paper_assets/retrain/attempt1.json                      # 1,015 s on this PC (20 cores, one GPU)
python paper_assets/scripts/run_eval.py --controllers rt1_final --levels 0,1,2,3 --episodes 300 --seed0 0 --workers 12 --out paper_assets/retrain/test_new.csv   # the one test run, 67 s
python paper_assets/retrain/scripts/make_after.py rt1_final                                                                       # episodes.csv, summary.*, charts
```

## What did not work, and caveats

- Cloning alone (`rt1_clone`) was poor (validation mean 0.231); the DAgger rounds did the work (0.525 after round 1).
- Two launches aborted with a `MemoryError` before any DAgger round (infrastructure, fixed; see `attempts.md`). No recipe change failed, because none was needed.
- The network is the same size as before and no wider or deeper network was tried. The improvement came with several changes at once (tuned teacher, more data, L3 states, more wind levels, lower last learning rate), so I cannot say which one mattered.
- The retrained controller still lands in only 0.300 of L3 episodes; like the expert, it is limited by the teacher (the pair report of 20 Oct says L2/L3 are limited by the expert). Step 2 (why the expert fails at L2/L3) is **not started**; it waits for your yes.
- Evaluation is deterministic (noise 0); DAgger rollouts used noise 0.1 on the executed action.
- The GPU was used for fitting, so a rerun may differ slightly in the last digits; the evaluator is deterministic for a given checkpoint.
