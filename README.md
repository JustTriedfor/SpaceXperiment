# SpaceXperiment

A 2D rocket-landing simulator for a Falcon 9-class booster, built from scratch in Python, together with two controllers that fly it: a hand-written **scripted expert** and a **neural-network controller** that learned from the expert by behaviour cloning and DAgger (PPO from scratch was tried and did not learn to land). The simulator uses its own rigid-body physics (engine, gimbal, grid fins, gas thrusters, drag, wind, burning fuel), and the project measures how often each controller lands safely at four wind levels. The measured landing success rates feed a Monte Carlo model of CO2e emissions per launch.

The project belongs to the paper **"Down to Earth: How Landing Success Probability Determines Expected Reflights and Life-Cycle CO2e per Launch of a Falcon 9-Class Booster"** by Firdaus Haryo Baskoro (SMA Semesta Semarang, Environmental category, GENIUS Olympiad).

This is a student project. It is **not affiliated with SpaceX**. SpaceX uses its own guidance software; nothing here is SpaceX code or SpaceX data, and the vehicle numbers are approximate public figures or my own estimates.

## Main results

A landing counts as a success only if **all four** hold: touchdown speed at most 3 m/s, tilt at most 0.15 rad, landing point within 10 m of the pad centre, and the fuel never runs out. Each cell is the number of successful landings out of 300 test episodes (seeds 0 to 299, no noise). The wind level is a scale: L1 = 5, L2 = 10, L3 = 15 m/s, and each episode draws its own wind within that scale.

| Controller | L0 (no wind) | L1 (5 m/s) | L2 (10 m/s) | L3 (15 m/s) |
|---|---|---|---|---|
| Final neural-network controller (`rt1_final`) | 300 | 259 | 161 | 90 |
| Scripted expert | 300 | 258 | 161 | 95 |
| Earlier neural-network controller (`sx_imitation_quiet`) | 298 | 204 | 113 | 43 |

These numbers are read from `paper_assets/retrain/summary.csv` (the per-episode rows are in `paper_assets/retrain/episodes.csv`). Tables with 95 % Wilson intervals and failure reasons are in `paper_assets/retrain/summary.md`.

## Limits (please read)

- **The simulator is simplified.** It is 2D (no sideways dimension), the vehicle numbers are approximate or estimated, and the wind is a simple model. It has **not been validated against real flight data**. The results do **not** predict how a real Falcon 9 lands.
- **The neural network is only as good as its teacher.** It imitates the scripted expert, so it cannot do better than the expert; at the hardest level (L3) both land in under one third of the episodes. Every failure of the expert at wind levels L1 to L3 missed the 10 m pad limit.
- **No sensor noise in training, and the controller is fragile to it.** In my test (`results/ablation_2026-10`) noise equal to 1 % of each observation's range, added at the controller input, gave **0 successes out of 300 at every level**. Landings already fail often at 0.25 % (for example 154 of 300 at L1). In that test 1 % means about 30 m of error on the height reading.
- **One seed set per level.** The test uses 300 episodes per level with seeds 0 to 299. Five retraining runs with different training seeds gave results within about 2 percentage points of each other, but they all start from the same earlier checkpoint.
- **The emissions model in `paper_assets/montecarlo` is a rebuild.** The original script was lost, so it was rebuilt from the paper's input table; the formulas were inferred and checked by reproducing the paper's Table 4.1 within 1 tonne. See that folder's README.
- **There is no git history before this upload.** The development dates in `paper_assets/progress` come from file times.

## Folder map

| Folder or file | What is in it |
|---|---|
| `src/sim/` | the simulator: physics, vehicle, wind and start states, the Gymnasium environment (`env.py`), the scripted expert (`expert.py`), two-lander mode, imitation pipeline |
| `src/training/` | reward, behaviour-cloning and DAgger code, PPO wrappers |
| `src/app/` | a browser viewer (`server.py`, `index.html`) that replays landings with telemetry panels |
| `src/utils/config.py` | all settings: wind levels, success rules, network size |
| `tests/` | 130 unit tests |
| `tools/expert_tuning/` | scripts used to tune the expert's constants |
| `models/checkpoints/` | the four small trained models used in the paper (about 0.6 MB each) |
| `assets/` | recordings of the first prototype (a Box2D lunar lander) |
| `paper_assets/` | everything behind the paper: evaluation scripts and CSV results (`scripts/`, `before/`, `retrain/`), failure analysis (`expert_failures/`), the emissions Monte Carlo (`montecarlo/`), the figures (`figures/`, `proposal/`) and the development history (`progress/`) |
| `results/ablation_2026-10/` | sensor-noise test and the five training-seed runs |
| `SPACEXPERIMENT.md`, `REWARD.md`, `RUNBOOK.md` | notes on the vehicle model, the reward, and how to run everything |

Training data and run logs (about 3 GB) are **not** in the repository. They can be regenerated by training (see below).

## Install

Python 3.11 is what the results were made with.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The pinned versions are the ones used for the paper. `torch==2.10.0` is the standard build; a GPU is optional.

## Run

Run everything from the project folder.

**1. Tests** (about 2 minutes)

```bash
python -m pytest -q
```

**2. Fly one landing** (the scripted expert, level L1, seed 122). Save the code below as `fly_one.py` in the project folder and run `python fly_one.py`.

```python
from src.sim.env import SpaceXEnv
from src.sim.expert import Expert
from src.sim.expert_data import _label

env = SpaceXEnv(level=1)
env.set_rules_stage(2)
obs, _ = env.reset(seed=122)
expert = Expert(); expert.reset()
done = False
while not done:
    obs, _, terminated, truncated, info = env.step(_label(expert, env, False))
    done = terminated or truncated
m = info["episode_metrics"]
print("success:", m["success"], "| touchdown speed (m/s):", round(m["speed_mps"], 2), "| pad offset (m):", round(m["dist_m"], 2))
```

To fly the trained network instead, load it with `PPO.load(str(resolve_checkpoint("rt1_final")), device="cpu")` (from `stable_baselines3` and `src.training.weights_io`) and use `model.predict(obs, deterministic=True)[0]` as the action.

**3. Evaluate controllers** (300 episodes per level, about 3 to 5 minutes with 8 workers; the `--out` file is written wherever you say, so the paper's CSV files are not overwritten)

```bash
python paper_assets/scripts/run_eval.py --controllers rt1_final,expert_tuned --levels 0,1,2,3 --episodes 300 --workers 8 --out my_results.csv
```

**4. Figures of the paper** (they write into `paper_assets/figures`)

```bash
python paper_assets/figures/make_fig4_method.py
python paper_assets/figures/make_figs_5_6_7.py
```

**5. Emissions Monte Carlo** (200,000 draws, a few seconds)

```bash
python paper_assets/montecarlo/emissions_mc.py
python paper_assets/montecarlo/make_outputs.py
```

**6. Browser viewer** (shows recorded landings; it needs a run folder in `logs/`, so first record one landing)

```bash
python paper_assets/scripts/make_live_episode.py --level 1
python src/app/server.py --port 8767 --run zz_paper_view --open
```

**7. Train again** (about 20 minutes to 1 hour; uses the earlier checkpoint as the start, writes into `data/`, `logs/` and `models/checkpoints/`)

```bash
python paper_assets/retrain/scripts/retrain_single.py --name my_run --config paper_assets/retrain/attempt1.json
```

By default the fitting step uses a CUDA graphics card. Without one, copy `attempt1.json` and add `"device": "cpu"` to it (slower). A tiny CPU version of this command (24 demonstrations, one round) was run to check that it works; the full run was not repeated for this README.

More commands, with the exact seeds and settings behind each paper result, are in `paper_assets/README.md`, `paper_assets/retrain/README.md` and `results/ablation_2026-10/README.md`.

## How to cite

Baskoro, F. H. (2026). *Down to Earth: How Landing Success Probability Determines Expected Reflights and Life-Cycle CO2e per Launch of a Falcon 9-Class Booster.* SMA Semesta Semarang (GENIUS Olympiad, Environmental category). Code: SpaceXperiment.

## License

MIT, see `LICENSE`.
