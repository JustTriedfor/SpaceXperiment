#!/bin/bash
# Job 4: five training seeds, one after the other (each about 17 min of training + 1 min of test).
cd /c/Users/firdo/rocketeer/SpaceXperiment
for s in 1 2 3 4 5; do
  python results/ablation_2026-10/scripts/train_seed.py --seed $s > results/ablation_2026-10/data/job4/run_seed$s.out 2>&1
done
echo finished > results/ablation_2026-10/data/job4/ALL_DONE
