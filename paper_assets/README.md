# paper_assets: landing-reliability evidence for the GENIUS Olympiad paper

Everything here is produced by new scripts in `paper_assets/scripts/`. No training or simulator code was changed. Run all commands from the project root (`C:\Users\firdo\rocketeer\SpaceXperiment`).

## Provenance

| item | value |
|---|---|
| commit hash | none: the project is not a git repository. Use the hashes below instead. |
| single-lander checkpoint | `models/checkpoints/sx_imitation_quiet.zip`, SHA-256 `a71c329b0344d0d7bc6a49f1d0bca77bd1596234a112bf56df3c5e4359324e7f` |
| pair checkpoint | `pair_safe_final_A.zip` SHA-256 `fde913c837ed6397346759c8506599ce871a3dee2aec5c8dc73d88eb9f474179`, `pair_safe_final_B.zip` SHA-256 `589138cb1fb059a46bec82d2aa594998a03e09a8289a6284aa5a356638bab0a1` |
| scripted expert | `src/sim/expert.py` (current tuned expert), SHA-256 `bc17fccb2bf768b9da5961cc209d39d9a889dcc21b8d8c29a579b5db89e569b4` |
| noise | 0. Policies run with `predict(deterministic=True)`, so there is no action noise. The only randomness is the seeded episode (spawn state and wind). |
| episodes | 300 per level per controller, levels L0-L3, seeds 0-299. 3600 episodes, 4800 CSV rows (the pair has two landers per episode). |
| runtime | `run_eval.py`: 160 s (second run, 12 workers). The first run took about 101 s. Both runs were identical on every required column. |

## Commands (one per output)

```bash
python paper_assets/scripts/run_eval.py --episodes 300 --levels 0,1,2,3 --workers 12   # episodes.csv (+ run_eval.log)
python paper_assets/scripts/make_rules.py                                              # success_rules.txt, wind_levels.txt
python paper_assets/scripts/make_summary.py                                            # summary.csv, summary.md
python paper_assets/scripts/make_charts.py                                             # chart_success_by_level.png, chart_touchdown_speed_hist.png
python paper_assets/scripts/make_live_episode.py --level 1                             # records the figure episode (seed 122) into logs/zz_paper_view/
python src/app/server.py --port 8791 --run zz_paper_view                               # separate viewer for the captures (leave port 8767 alone)
python paper_assets/scripts/capture_ui.py --port 8791                                  # fig3_1, fig3_2, fig4_1, fig4_2, fig4_2b
```

## Wind (please check against the paper)

The wind levels are not constant means of 0/5/10/15 m/s. In `src/sim/spawn.py` each episode draws a signed mean `U(-1,1) * 0.6 * wind_power` and adds gusts of amplitude `0.4 * wind_power`. So 0/5/10/15 is a scale. The realised mean per episode is in `wind_mean_mps`; the statistics are in `wind_levels.txt`. The success limits are in `success_rules.txt`: speed <= 3 m/s, tilt <= 0.15 rad, pad offset <= 10 m, no propellant exhaustion, ended at rest.

## Headline numbers (success rate, 95 % Wilson interval)

| controller | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| sx_imitation_quiet (n=300) | 0.993 [0.976, 0.998] | 0.680 [0.625, 0.730] | 0.377 [0.324, 0.433] | 0.143 [0.108, 0.188] |
| pair_safe_final, per lander (n=600) | 0.998 [0.991, 1.000] | 0.875 [0.846, 0.899] | 0.588 [0.549, 0.627] | 0.350 [0.313, 0.389] |
| pair_safe_final, joint (n=300 episodes) | 0.997 [0.981, 0.999] | 0.860 [0.816, 0.895] | 0.573 [0.517, 0.628] | 0.320 [0.270, 0.375] |
| expert_tuned (n=300) | 1.000 [0.987, 1.000] | 0.860 [0.816, 0.895] | 0.537 [0.480, 0.592] | 0.317 [0.267, 0.371] |

Clean trials needed (95 % one-sided): p >= 0.99 needs 299 (exact) or 300 (rule of three); p >= 0.9963 needs 809 or 811. Achieved: only `expert_tuned` at L0 is clean (300/300, lower bound 0.990, supports 0.99). `pair_safe_final` L0 has one failure in 600 landers (lower bound 0.992), but its landers are not independent, so the joint row (0.984) is the honest one. Nothing reaches 0.9963. Full tables are in `summary.md`.

## UI figures

All were captured at 1920x1080, device scale 2, dark theme, from one recorded landing: `sx_imitation_quiet`, level 1, **seed 122** (success, 2.416 m/s, 2.379 m offset, 444 frames; the success closest to the median speed and offset of that level).

- `fig3_1_interface_full.png`: the whole UI about 28 m above the pad (t = 19.7 s), all panels visible. Panels were moved by script so none overlap. The header shows the viewer's placeholder text ("training step 1.0k", "exploring policy"), because the episode is replayed through a throw-away run folder.
- `fig3_2_recording_panels.png`: landing cam, flight display, telemetry (including forces) and fuel profile, at the same frame.
- `fig4_1_landing_scatter.png`: fed from `episodes.csv` (all 1200 `sx_imitation_quiet` touchdowns, colour per level; the viewer normally keeps only the last 400). A white ring marks the newest dot. Filled = ended at rest.
- `fig4_2_descent_phase_portrait.png` and `fig4_2b_decision_boundary.png`: the same episode at the last frame, so the full path, brake limit and envelope show.

Not captured as in the viewer: the decision boundary's two footnotes overlap each other inside the viewer at any width, so `fig4_2b` is cropped just above them. Fixing that needs a viewer code change, which I did not make.

## Surprising or worth checking

1. The trained single-lander controller falls off quickly with wind: 0.993 at L0 but 0.143 at L3, almost all `too_fast`. The tuned expert beats it at every level, and `pair_safe_final` beats it per lander.
2. Failure modes differ. The imitation controller fails on speed only. The expert and the pair controller also fail on `off_pad`: 40/71/54 for the expert at L1/L2/L3, 66/142/120 for the pair.
3. Out-of-fuel and tilt failures are zero everywhere, so those two rules never decide a result here.
4. The scatter's "green box" counter says 661, while the success count is 658. The viewer's box tests offset and speed only, not tilt or rest.
5. The pair's two landers share wind and start, so per-lander n overstates the evidence.
