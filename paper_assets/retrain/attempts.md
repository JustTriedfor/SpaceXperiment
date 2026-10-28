# Recipe attempts (every attempt is listed; the choice is made on the validation seeds only)

Validation seeds 100000-100199 (200 episodes per level, same evaluator as the paper). The test seeds 0-299 were **not** run for any attempt until the attempt was chosen, once, at the end.
Reference on the validation seeds (not training data): scripted expert 1.000 / 0.840 / 0.465 / 0.235 for L0 / L1 / L2 / L3; old `sx_imitation_quiet` 0.990 / 0.655 / 0.360 / 0.145.
Stop rule (decided before the run): stop when the best validation round is within 5 percentage points of the expert at L1, L2 and L3 (i.e. at least 0.790 / 0.415 / 0.185).

## Attempt 1 (`rt1`, the only recipe attempt needed): pair-style recipe with the tuned expert, more data, L3 states, lower final learning rate

Config: `attempt1.json` (all other values are the script defaults, printed at the top of `logs/rt1.log`). Start from `sx_imitation_quiet` (as the pair did); 800 expert demonstrations (734 kept, 340,475 steps) from the current tuned expert on levels 0,1,1,2,2,3; behaviour cloning 40 epochs at lr 1e-3; 8 DAgger rounds of 180 episodes on levels 0,1,1,2,2,3,3 (executed noise 0.1); 20 epochs per round at batch 1024 (lr 5e-4, lr 1e-4 in the last two rounds); weight 5 on states below 400 m; GPU. Total 1,015 s.

| model | validation L0 | L1 | L2 | L3 | mean | training steps in the fit |
|---|---|---|---|---|---|---|
| `rt1_clone` (cloning only) | 0.400 | 0.255 | 0.180 | 0.090 | 0.231 | 340,475 |
| `rt1_dagger_1` | 0.915 | 0.645 | 0.360 | 0.180 | 0.525 | |
| `rt1_dagger_2` | 0.980 | 0.805 | 0.455 | 0.215 | 0.614 | |
| `rt1_dagger_3` | 0.990 | 0.815 | 0.475 | 0.225 | 0.626 | |
| **`rt1_dagger_4`** (chosen) | 1.000 | 0.840 | 0.485 | 0.225 | **0.638** | 662,965 |
| `rt1_dagger_5` | 1.000 | 0.840 | 0.480 | 0.220 | 0.635 | |
| `rt1_dagger_6` | 1.000 | 0.840 | 0.470 | 0.235 | 0.636 | |
| `rt1_dagger_7` (lr 1e-4) | 1.000 | 0.840 | 0.470 | 0.220 | 0.633 | |
| `rt1_dagger_8` (lr 1e-4) | 1.000 | 0.845 | 0.470 | 0.225 | 0.635 | 986,663 |

Result on validation: the stop rule was met from round 3 (0.815 / 0.475 / 0.225). The chosen round is the one with the highest validation mean (round 4). Rounds 4-8 differ by less than the validation sampling error (about 3 points), so the choice among them is not meaningful, which is why nothing was tuned further. No second recipe was tried.

## Runs that did not count (infrastructure, no recipe change)

Two launches of attempt 1 stopped after the cloning step with a `MemoryError` in the evaluation worker pool (the parent process held the GPU context). Fix: the fitting step runs in a child process (`scripts/fit_worker.py`). All logs and data of the aborted launches were deleted and the attempt was restarted from the beginning with the same configuration; the reported run is the third launch. A tiny smoke test (24 demonstrations, 1 round) was run first and deleted.
