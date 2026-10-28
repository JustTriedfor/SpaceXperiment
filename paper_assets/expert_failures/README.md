# Why the scripted expert fails at L2 and L3 (step 2; the expert and simulator were not touched)

Data: the expert's 1,200 test episodes (seeds 0-299, L0-L3) in `paper_assets/before/episodes.csv`, plus the start state that `env.reset(seed)` gives for each of them (nothing was flown again). Script: `python paper_assets/expert_failures/analyze.py`; all tables are in `failure_table.md`, per-episode rows in `failure_by_start.csv`.

## Failure reasons per level (one label per failed episode)

| level | failures / 300 | too_fast | tilted | off_pad | out_of_fuel | crashed_other |
|---|---|---|---|---|---|---|
| L0 | 0 | 0 | 0 | 0 | 0 | 0 |
| L1 | 42 | 2 | 0 | 40 | 0 | 0 |
| L2 | 139 | 68 | 0 | 71 | 0 | 0 |
| L3 | 205 | 151 | 0 | 54 | 0 | 0 |

The label is the first matching rule in the order out_of_fuel, too_fast, tilted, off_pad, so `tilted` is always hidden behind `too_fast` or `off_pad`. Counting every rule an episode broke gives a clearer picture:

| level | failures | broke off_pad | broke too_fast | broke tilted | broke out_of_fuel |
|---|---|---|---|---|---|
| L1 | 42 | **42** | 2 | 1 | 0 |
| L2 | 139 | **139** | 68 | 57 | 0 |
| L3 | 205 | **205** | 151 | 107 | 0 |

## Main cause

**The expert does not get the stage over the pad.** Every failed episode at L1, L2 and L3 landed (or ended) more than 10 m from the pad centre; speed, tilt and fuel are not what separates success from failure in most of them.
- Median miss of the failures: 18.0 m (L1), 39.2 m (L2), 83.3 m (L3), against 1.5 / 2.4 / 3.7 m for the successes.
- Fuel is not the limit: no episode ran out of propellant, and failures end with about as much propellant left as successes (L2: 2,014 kg vs 1,950 kg; L3: 1,934 kg vs 1,854 kg of 4,500 kg).
- The `too_fast` failures at L2 and L3 are also off-pad (68 of 68 and 151 of 151), and most of the L3 ones also tilted past 0.15 rad and did not end at rest, i.e. they are crashes, not gentle landings that were slightly too fast. At L3 the median touchdown speed of failures is 7.9 m/s, while the successes land at 2.1 m/s.
- The miss follows the **start geometry more than the wind**: 76-83 % of failures land on the same side of the pad as the start offset, but only 52-55 % on the side the wind pushes (no better than chance). Success falls with the start offset (L2: 0.67 / 0.54 / 0.40 for the lowest / middle / highest third of |start x|; L3: 0.43 / 0.29 / 0.23), whereas it does not fall steadily with the wind (L2: 0.45 / 0.63 / 0.53; L3: 0.26 / 0.40 / 0.29). The level table also raises the start offset (50 / 120 / 220 / 350 m scale), entry speed and entry angle together with the wind, so "L3 is harder" mostly means "L3 starts farther to the side and faster", not only "L3 is windier".

## What this does and does not show

- Association, not proof: start offset, sideways speed and fall speed move together inside a level, so these tables cannot say which of them the expert cannot handle. They also cannot say at which altitude the sideways error becomes uncorrectable.
- Consistent with the note in `logs/pair_safe_report.md` (20 Oct): "Levels 2 and 3 are limited by the teacher: more sideways speed than a thrust-tilted stage can remove in the burn". This analysis supports "limited by lateral reach" but did not test that mechanism.
- What would settle it (not done, needs flights with traces and no change to the expert): log lateral error and sideways speed against altitude for failed and successful seeds at L2/L3, and check whether the expert's steering saturates (gimbal, fins, tilt limit) before the burn ends.
- Consequence for the retrained controller: it imitates this expert, so its L2/L3 ceiling (0.537 / 0.300) is the expert's (0.537 / 0.317). Raising it needs a better teacher, not more imitation.
