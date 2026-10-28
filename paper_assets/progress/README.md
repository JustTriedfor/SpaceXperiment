# Development evidence for SpaceXperiment (GENIUS Olympiad)

**Read this first: the project has no git history.** `C:\Users\firdo\rocketeer` and `C:\Users\firdo\rocketeer\SpaceXperiment` are plain folders (the `.gitignore` exists, there is no `.git`). The only `.git` folders on this machine belong to unrelated projects (`claude-usage`, an empty `.openclaw/workspace`). No other copy, backup or remote of the project was found, and no folder named "lunar lander" exists apart from the prototype described below. So everything here is dated by **file-system times** (creation and last-modified), there are **no commit hashes**, and old versions cannot be checked out. Nothing was edited or recreated; where an old version cannot be run, it is listed under "Not reproduced".

## What is in this folder

| file | what it is |
|---|---|
| `file_history.txt` | every source file of both folders with creation and modification time (A: Rocketeer prototype, B: SpaceXperiment), plus every training-run folder with its episode count and last write time (C) |
| `milestones.md` | 15 milestones in date order, each with its evidence files, a quote from the file, and what can be seen or measured |
| `timeline.png` | the milestones on a date axis |
| `img_01..03_*.png` | real captures of three versions (below) |
| `fig3_1_early_versions.png` | the three captures side by side with captions (2000 x 700) |
| `existing/` | the three recordings that already existed in the project (not edited) |
| `train_custom_s0_m2.png`, `train_custom_s0_staged8.png`, `train_custom_s0_quiet.png`, `train_custom_s0_pair.png`, `training_runs.md` | learning curves from real logs, and which file each came from |
| `scripts/` | the scripts that made everything |

## Images and the commands that produced them

No commit exists, so each image is labelled with the date of the files it shows.

| image | version / date | what it shows | how it was made |
|---|---|---|---|
| `img_01_2026-10-18_lunarlander-prototype-gif-frame.png` | Rocketeer prototype (gymnasium Box2D LunarLander-v3), 18 Oct 2026 13:52 | frame 43 of 631 of `assets/custom_l0_m2.gif`, native 600x400, taken from the existing recording | `Image.open(...).seek(43)` (frame chosen as the first one with the lander about 80 px above the pad) |
| `img_02_2026-10-19_rocketeer-box2d-viewer.png` | Rocketeer browser viewer, files last modified 19 Oct 2026 18:03 | lander at touchdown (0.34 m/s), telemetry, phase portrait, nadir cam | viewer served from a temp copy of `rocketeer/src` on port 8793 showing run `custom_s0_staged8`, episode 4428 (environment 6; the file records no seed); `python paper_assets/progress/scripts/capture_versions.py proto <out.png>` |
| `img_03_2026-10-21_spacexperiment-current.png` | SpaceXperiment as it is now (21 Oct 2026) | stage in flight about 50 m up, plume, landing cam, telemetry | `python paper_assets/scripts/make_live_episode.py --level 1 --seed 122`, then `python src/app/server.py --port 8791 --run zz_paper_view`, then `python paper_assets/progress/scripts/capture_versions.py current <out.png>`; **seed 122**, `sx_imitation_quiet`, level 1 |

img_02 and img_03 are 1280x720, headless Chromium (Playwright), default panels, paused at 80 % and 86 % of the recorded episode. The two viewers are not the same episode type (img_02 comes from a training run of the prototype; its network panel says "no neural data" because that episode file has no activations), so they are not the same seed. In `fig3_1_early_versions.png` image (a) is enlarged from 600x400.

Existing recordings in `existing/` (copied with their original modification time, not edited): `custom_l0_m2.gif` (2026-10-18 13:52, 631 frames), `custom_l0_locked.gif` (2026-10-18 13:55, 363 frames), `custom_l0_m2_progress.png` (2026-10-18 13:52, the prototype's own progress plot). Copies of the same three files are also in `SpaceXperiment/assets/`.

## Training evidence

Plotted from `episodes.csv` of each run folder, one point per fixed bin of training steps, no moving average (details in `training_runs.md`; dashed lines are the steps at which the curriculum level or rules stage changed, from `levels.csv` / `stages.csv`):

| run | project | what the curve shows |
|---|---|---|
| `custom_s0_m2` (3,906 episodes, 18 Oct 13:49) | prototype | strict success stays 0; median touchdown speed falls from about 9 to about 2.5 m/s over 500k steps; the share of episodes that reach the ground dips to about 0.1 at 70k steps and then rises to 1.0 |
| `custom_s0_staged8` (10,627 episodes, 18 Oct 20:24) | prototype, staged rules | median touchdown speed falls from 19 to about 3 m/s; soft landings reach up to about 0.1 of the episodes in a bin; strict success stays near 0 |
| `custom_s0_quiet` (7,000 episodes, 20 Oct 08:49) | SpaceXperiment | PPO fine-tuning of the imitation network: soft-landing share **falls** from 0.33 to about 0 over 3 M steps while median touchdown speed rises. This is the degradation the hand-off note describes |
| `custom_s0_pair` (1,646 episodes, 21 Oct 11:53) | SpaceXperiment, pair mode | soft-landing share rises to about 0.4 around 0.25 M steps, then falls to about 0.1 by 1 M steps. This run was still writing when it was read |

So the logs show the imitation controller and the expert, not PPO from scratch, as the source of reliable landings. Do not describe the curves as steady learning.

## Numbers

| | value |
|---|---|
| first file | 2026-10-18 12:57 (`.gitignore`); first source file 13:14 |
| last source modification | 2026-10-21 07:25 (this evidence folder and `paper_assets/` were added on 21 Oct) |
| commits | none (not under version control) |
| Rocketeer prototype | 31 Python files / 5,448 lines, 1 HTML-JS-CSS page / 1,914 lines, 2 Markdown / 430 lines |
| SpaceXperiment | 57 Python files / 8,347 lines, 1 HTML-JS-CSS page / 3,311 lines, 4 Markdown / 917 lines (without `paper_assets/`, logs, models) |
| tests | prototype 168 collected, **168 passed** (60 s); SpaceXperiment **130 passed** (119 s). Both run on 21 Oct 2026 with `python -m pytest -q` in their folder |

Line counts are plain line counts (`cloc` is not installed); empty lines and comments are included. Only these file types were counted: `.py`, `.html`, `.md`, `.ps1`, `.bat`, `.ini`, `.txt`.

## Not reproduced

- **Intermediate versions.** Without commits there is no snapshot of the code at any earlier time, so only the state on disk can be run. The prototype folder is its state at 19 Oct (last edit), not its first version. No "mid-development" screenshot of SpaceXperiment exists or could be made. The listed milestones between 18 and 21 Oct are therefore shown by file evidence, not by pictures.
- **The prototype's own physics run.** The prototype needs gymnasium's Box2D environment; I did not run it. Image 01 is a frame from its existing recording instead.
- **Panel-by-panel dates.** The viewer is one HTML file, so when each panel was added cannot be recovered. The only verifiable statement: the 19 Oct prototype page already has the telemetry, nadir/landing cam, phase portrait, flight recorder, decision boundary, fuel profile, landing scatter and flight display; it lacks the partner cam, the Dual land mode, the Pipeline tab and the airflow picture.
- **Timestamps caveat.** Files copied into `SpaceXperiment/` at 2026-10-19 21:02 show that time as creation time and keep their old modification time, so the milestone dates for those files use the modification time. A restore or copy of the folder would change times.

## For the paper (about 100 words, verified items only)

The simulator was developed between 18 and 21 October 2026 (file timestamps; the project is not under version control). A first prototype wrapped the Box2D lunar lander of the gymnasium library with fuel, tilt and stall rules, four wind levels, PPO training and a browser viewer. On 19 October the Box2D dynamics were replaced by our own 2-D rigid-body physics of a Falcon 9 first stage, with wind shear and real-scale re-entry starts. On 20 October a scripted expert controller, imitation learning and a two-lander pad A/B mode were added. 130 automated tests pass, and the controllers were evaluated on 3,600 seeded episodes.
