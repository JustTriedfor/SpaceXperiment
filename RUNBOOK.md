# SpaceXperiment runbook

How to start, run, watch, shut down and troubleshoot **SpaceXperiment**: the real-scale Falcon 9 landing version of the project (own 2-D physics, 3 km start at up to ~300 m/s, drag, g = 9.81, gimballed main engine + grid fins + two RCS systems).
Project folder: `C:\Users\firdo\rocketeer\SpaceXperiment` (app on **http://127.0.0.1:8767**). The original Rocketeer one folder up is a separate project with its own runbook and its own app on port 8765; the two can run side by side. Read `SPACEXPERIMENT.md` for what the vehicle, spawn, rules and calibration are, and `REWARD.md` (sections 12-13) for the reward numbers.

Everything below is PowerShell unless noted.

---------------------------------------------------------------------------------------------------

## 0. Simple / Advanced and the Startup tab (read this first)

**The switch.** Under the Training card there is an **Interface: Simple | Advanced | Showcase | Showcase Recorder** switch (Simple and Advanced are remembered in your browser; the page opens in Simple unless you chose Advanced before). **Showcase** opens the replay display for a public stand (section 6b); it is never remembered, so the normal page always opens as before. **Showcase Recorder** opens the screen that makes the Showcase's recordings (also section 6b); it is a plain page that shows no picture and is not remembered either.
- **Simple**: only four tabs: **Live**, **Startup**, **Settings** and **Models**. Everything is a slider or a drop-down menu, every slider stops at a safe value, and menus only offer things that exist and fit (for example only brains of the right kind). Free-typed numbers and names are gone: new models are named for you. Switching to Simple pulls any unsafe value you had typed in Advanced back into the safe range (learning speed at most 3 x 10^-4, at most 8 worlds, at most 5 M steps, sweep off, ...).
- **Advanced**: every tab and every setting, exactly as before (Brain, Parameters, Pipeline, Checkpoints, Data, Tests, all the chips and panels), plus Startup.
Both modes edit the same underlying settings, so you can switch whenever you like.

**Settings (Simple)**: What to fly (one lander / two landers one brain / two landers two brains), which brain to start from (only brains of that kind, best-tested first), training length (0.1-5 M steps), worlds at once (1-8), learning speed (6 safe steps from 1x10^-5 to 3x10^-4), random exploration (none / a little), starting difficulty, **re-entry angle** and **trajectory variation** (below), gradual rules, and for two landers: pad distance (300-3000 m), starting height (2000-5000 m), how different the two starts are, and the "together" tolerance (1-5 s). Then press **Start training**; the line next to the button says which run it writes to.

**Models (Simple)**: pick a model; make a quiet copy (noise slider 0.02-0.15), a pair of two brains, or a dual copy; test it (10-50 episodes, levels 0 / 0-1 / 0-2); continue it with PPO; move it to the trash. Names are made for you (the old name + _pair / _dual / _quiet).

### Touchdown: 2.5 s to straighten up

A lander that touches down a little tilted, or on one leg first, no longer freezes the instant it touches: its legs give (the viewer shows them shortening), the engine shuts down, and it has **2.5 s** to settle upright; it is judged when it has been upright and nearly still for 0.6 s, or when the time is up. Too fast / too sideways / tilted more than 0.45 rad at the first touch is still a crash at once. The time is the **"Time to straighten up after touchdown"** slider in Settings (Simple) / *Time to settle after touching down* in Parameters (Advanced); 0 restores the old rule. Details: SPACEXPERIMENT.md, "Touchdown". Only runs started after this change use it.

### Re-entry angle and trajectory variation

Two settings decide what the start of every episode looks like (Settings and Startup in Simple; *Re-entry angle* and *Trajectory variation* in Parameters > Start conditions in Advanced; they also reach the expert pipeline's demonstrations, so the data matches the training):
- **Re-entry angle** = the flight-path angle below the horizontal when the lander starts, from shallowest to steepest: **extreme** (35-48 deg: a long sideways glide, thousands of metres to cover), **intermediate** (48-62), **SpaceX** (62-74, about what a real booster's landing-burn entry looks like), **normal** (each level's own range: level 0 is 72-85 deg ... level 3 is 42-72; this is the default and what every older run used) and **straight down** (87-90).
- **Trajectory variation** (0-100%): how much the entry angle and the entry speed change from episode to episode. 100% uses the whole range (the old behaviour), 50% a band half as wide around the middle, 0% always the middle of it: every episode is the same path.
The scripted expert lands at level 0 in 90% of extreme starts, 95% intermediate, and 100% of the others (40 starts each).

### The Startup tab (both modes): from nothing to a trained pair in three choices
1. **What do you want to fly?** One lander / two landers with one shared brain / two landers with two separate brains (recommended). The **re-entry angle** and **trajectory variation** (below). For two landers: the distance between the pads (85-3000 m).
2. **How should they get their first idea of what to do?**
   - *Copy a brain I already have (instant)*: pick a single-lander brain (the best-tested one is preselected); for two landers it is copied into a pair (or one bigger dual brain). They then already land like the original.
   - *Teach them with the expert first*: runs the whole expert pipeline for you (scripted expert -> cloning -> DAgger -> test) with a **How thoroughly** slider: Quick (about 12 min for one lander, 18 for two), Standard (about 35 / 50 min), Thorough (about 80 / 120 min). If you pick a brain it is the starting point and the cloning step is skipped (cloning on little data can only make it worse); with "None" it starts from scratch and clones first.
   - *No head start*: the landers start blank (not recommended).
3. **Then train them**: tick *start training automatically* and set the length, worlds at once, learning speed and starting difficulty. When the head start is done, training starts **by itself** from the finished brain, even if you closed the page. In Advanced you can also give the run a name and let it replace old outputs of that name.

Press **Start**. The *Plan* box says in plain words what will happen, and *Progress* shows each stage (head start, then training), the practice rounds with their scores and the final test. **Stop everything** stops the pipeline and any training. **Watch it live** switches to the Live tab (the viewer follows the new run). The state is kept in `logs\startup.json`, so a refresh just picks it up again.
Rules: only one pipeline at a time; training will not start if any training is already running on the machine (even a Rocketeer run: the Stop button kills those too, so be careful); the pipeline and the training use the CPU together.

---------------------------------------------------------------------------------------------------

**Starting from a pair you already have:** with *Two landers, two separate brains* and *Copy a brain I already have*, the brain list also shows your **tested pairs** (marked PAIR (two brains), for example `pair_safe_final`, tested 88%). The best tested one is picked by default. A pair is used as it is: nothing is copied, and training continues from both of its brains. Pairs are not offered for one lander, one shared brain, or the expert option.

## 1. Start up

1. **Open PowerShell in the project folder.**
   ```powershell
   cd C:\Users\firdo\rocketeer\SpaceXperiment
   ```
2. **(Optional) check nothing is left over.** The output should be empty (or only things you know about, such as the Rocketeer app on 8765).
   ```powershell
   Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'train.py|sweep.py|server.py|finetune.py|dagger_loop|expert_data' } | Select-Object ProcessId, CommandLine
   ```
3. **Start the control panel** (opens the browser at http://127.0.0.1:8767):
   ```powershell
   .\run_app.ps1
   ```
   If scripts are blocked: `powershell -ExecutionPolicy Bypass -File .\run_app.ps1`, or double-click `run_app.bat`. Another port: `.\run_app.ps1 -Port 8768`. Leave the window open: it is the server. (Training runs as a separate process, so closing the server does not stop a run.)
4. **(Optional) run the tests.** Tests tab, *Run tests* (should say PASSED), or `.\run_tests.ps1`. 120 tests, about 2 minutes.
5. **Set the run up** in the **Parameters** tab (changed values are outlined in blue), then press **Start training** (top-right panel). The chip next to the title says *training*, the progress bar moves, and the Live tab fills with boosters within 10-30 s.

Starting a run from the command line instead (same defaults, no web form):
```powershell
python src\training\train.py --reward custom --seed 0 --total-steps 3000000 --n-envs 4 --start-level 0 --tag _cli
python src\training\train.py --reward custom --seed 0 --total-steps 3000000 --n-envs 4 --init-from sx_imitation_quiet --tag _cli
```
The app passes the rest of its form (learning rate, entropy, rules stage, ...) in the environment variable `ROCKETEER_OVERRIDES`; a command-line run uses the defaults in `src\utils\config.py` unless you set it, for example
`$env:ROCKETEER_OVERRIDES='{"config":{"RULES_STAGE_START":0,"PPO_KWARGS":{"learning_rate":0.0003,"ent_coef":0.0}}}'`.
Or drive the app's API (what the Start button does):
```powershell
curl.exe -s -X POST http://127.0.0.1:8767/api/train/start -H "Content-Type: application/json" -d "{\"n_envs\":4,\"start_level\":0,\"staged\":\"on\",\"init_from\":\"sx_imitation_quiet\",\"learning_rate\":0.0003,\"ent_coef\":0,\"tag\":\"_quiet\"}"
curl.exe -s http://127.0.0.1:8767/api/train/status
curl.exe -s -X POST http://127.0.0.1:8767/api/train/stop -H "Content-Type: application/json" -d "{}"
```

## 2. Shut down

1. **Stop training**: *Stop* in the Training panel (or the stop call above). It kills the trainer and its workers; the chip goes back to *idle*. Skip it if the run finished by itself (log says "done").
2. **Check it is gone** (step 1.2 command).
3. **Stop the server**: Ctrl+C in the window running `run_app.ps1`.
4. Close the browser tab.

**Clean up leftovers** (crash, closed window, a second copy of a script still running):
```powershell
Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'train.py|sweep.py|finetune.py|dagger_loop|expert_data|multiprocessing' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```
This also stops the Rocketeer trainer if it is running. Leave out `server.py` so the apps keep running, or add it to stop them too.

---------------------------------------------------------------------------------------------------

## 3. Where things are saved

| What | Where |
|---|---|
| Per-episode metrics, level and rules-stage changes | `logs\<run>\episodes.csv`, `levels.csv`, `stages.csv` |
| Training console output | `logs\<run>.out` |
| Settings a run used | `models\configs\<run>.json` |
| Checkpoints (trained weights) | `models\checkpoints\` (`sx_imitation.zip` = the cloned expert, `sx_imitation_quiet.zip` = the same with low exploration noise) |
| Step-level experience (for export / cloning) | `data\trajectories\<run>\` |
| Calibration results | `models\configs\calibration_spacexperiment.json` |
| Old data / checkpoints from before the 18-input observation | `data\_archive_old_expert\`, `models\_archive_obs14\` (not loadable by the current code) |

Run name = `<reward>_s<seed><tag>` (for example `custom_s0_quiet`). Starting a run with the same name overwrites its logs: change the **run tag** to keep old results.

---------------------------------------------------------------------------------------------------

## 4. The Parameters that matter here

- **Parallel environments**: workers feeding one shared policy. 4 or 8 is plenty. The 256 x 256 network is about 830 steps/s with 8 CPU environments.
- **Staged rules** (on): stage 0 = tilt limit, slow-approach charge, crash and landing scores; stage 1 adds the descent clock and the stall rule; stage 2 adds the ignition limit (3). They switch on as the agent survives.
- **Start level**: 0-3 = 35% / 60% / 80% / 100% of the full entry speed, steeper to shallower entry angle, more wind. The curriculum moves up when the gate passes **or after "Max steps per level" (default 500k) even if it has not**: watch for that (see 7).
- **Learning rate / entropy / n_steps / batch**: PPO tab. Default learning rate 3e-4. For a run that starts from a clone use 1e-4 to 3e-4 and **entropy 0**.
- **Continue from checkpoint**: weights of an existing model (shared-policy runs only). The optimiser and step counter start fresh.
- **Device**: leave on `cpu`. The GPU is about 2x slower for this network (every simulation step needs the policy, so the CPU-GPU round trips cost more than the matrix work saves).
- **Start conditions**: spawn height (3000 m is the real landing-burn start), entry speed range, speed scale. Lower the speed scale to make everything easier.
- **Reward**: the *Custom reward* section; numbers in REWARD.md section 13.

---------------------------------------------------------------------------------------------------

## 5. Starting from an expert (imitation, DAgger)

The booster is hard to learn from scratch, so there is a scripted expert (`src\sim\expert.py`) and a pipeline that clones it into the network. Run these from the project folder; **run only one copy of each script at a time** (two copies write the same folders and give confusing results).

| Step | Command |
|---|---|
| How good is the expert? | `python -m src.sim.eval_expert --n 150` (success and failure reasons per level; level 0 should be ~100%, level 1 ~75%) |
| Expert demonstrations (+ an untrained network to clone into) | `python -m src.sim.expert_data --episodes 400 --out expert_sx --noise 0.12 --levels 0,0,1,1,2 --make-init sx_init` (~40 s) |
| Behaviour cloning | `python src\training\finetune.py --checkpoint sx_init --data expert_sx --out sx_expert_clone --epochs 40 --lr 0.001 --batch 512 --log-std -1.6 --low-weight 5` (~4 min) |
| DAgger rounds (policy flies, expert labels, retrain) | `python -m src.sim.dagger_loop` (6 rounds, ~15 min each; edit the names at the top of the file) |
| Fly a checkpoint deterministically | `python src\sim\eval_policy.py sx_imitation 100` |

Reference result (`sx_imitation`, 100 episodes per level, deterministic): level 0 **100/100** soft landings, level 1 70/100, level 2 45/100.

**Exploration noise.** PPO adds Gaussian noise to every action (std 0.2 by default). The cloned policy lands 30/30 with no noise and with half the noise, but only ~3/30 at std 0.2, because the engine sits just above its off-threshold during the final flare and noise switches it off. So continue PPO from **`sx_imitation_quiet`** (std 0.08), not from `sx_imitation`. To make another quiet copy:
```powershell
python -c "import torch; from stable_baselines3 import PPO; m=PPO.load('models/checkpoints/sx_imitation.zip', device='cpu'); m.policy.log_std.data.fill_(-2.5); m.save('models/checkpoints/my_quiet.zip')"
```

**Observation channels.** The network sees 18 inputs: the 14 base ones plus x, height, vx and vy again on a square-root scale. They matter: on the linear scale the last 20 m is 0.007 of the input range and the clone never learns the flare.

---------------------------------------------------------------------------------------------------

## 5b. Dual land (start profile "dual")

One environment, **two landers and two pads**. Both landers start together 4,000 m up on the **same predetermined path** (one draw of entry speed, angle, side and wind per episode, given to both, plus a small independent jitter), their pads are `Pad spacing` apart (default 1,000 m; the landers start that far apart too), and the goal is to **land together**. When one lander is down it stays where it is (engine off) until the other one is down too; only then does the next episode start for both. Pick it in Parameters > Start conditions > *Start profile = dual*; the **Dual land** parameter group then applies:

| Parameter | Default | Meaning |
|---|---|---|
| Pad spacing (m) | 1000 | distance between the pad centres and between the landers' starts (100-6000) |
| Spawn height (m) | 4000 | both start this high above their pads |
| Start differences | 0.3 | 0 = exactly the same path; 1 = entry speed +-3%, tilt +-1 deg, spin and 8 m of position, independently per lander |
| Landing together: tolerance (s) | 2.0 | touchdown-time gap up to which they count as landing together (joint success needs it) |
| Landing together: bonus | 1000 | paid when both survive: bonus x (1 - gap / tolerance) |
| Height-gap shaping (points per m) | 0.5 | reward for closing the height gap between the two while both are airborne (potential shaping, no drift) |

**How it is steered.** One policy controls both: 40 inputs (lander A's 18 observations, lander B's 18, then height gap, fall-speed gap, A down, B down) and 10 outputs (A's five actions, B's five). Each lander's observation is relative to its own pad, so the single-lander skills carry over. The network is 512 x 512. Reward = both landers' normal rewards + the two landing-together terms (`term_sync_shape`, `term_sync_bonus` in episodes.csv). **Joint success** = both soft, on their pads and within the tolerance; `sync_dt_s` in episodes.csv is the touchdown gap (empty if one was lost). The episode's `end` is the worse of the two. Levels and the staged rules work as before.

**Starting a dual run** needs a dual checkpoint (a single-lander one does not fit: the trainer says so and stops). Build one from any single-lander checkpoint: it starts as two independent copies of it side by side, with zero weight on the four extra inputs:
```powershell
python -m src.sim.dual_init --from sx_imitation_quiet --out dual_imitation_quiet     # already in models\checkpoints
python -m src.sim.eval_dual dual_imitation_quiet 30 --levels 0,1                       # fly it: both at rest, joint success, landing-time gap
curl.exe -s -X POST http://127.0.0.1:8767/api/train/start -H "Content-Type: application/json" -d "{\"start_profile\":\"dual\",\"n_envs\":2,\"init_from\":\"dual_imitation_quiet\",\"learning_rate\":0.0001,\"ent_coef\":0,\"tag\":\"_dual\"}"
```
Reference (from `sx_imitation`, no training): level 0 = 30/30 joint successes, median landing gap 0.05 s; level 1 = 19/30 (the single-lander clone's level-1 rate). A 30k-step PPO run from the quiet dual checkpoint at learning rate 5e-5 kept 48 of 50 episodes as joint successes.

**In the viewer:** both landers are always drawn (A in the run's colour, B in a shifted hue) with their own plumes, smoke and trails, over two labelled pads (PAD A, PAD B). The auto camera frames both landers and both pads. The panels, the terminal, the follow cam and the telemetry follow the **focused** lander: key **B**, the **Focus** button under the picture, or a click on a lander switches (the neural-network panels still show the single dual network). At the end of an episode a note says how far apart they landed. The HUD line says DUAL LAND with the spacing. Graph panels that rebuild the network's inputs by hand (decision boundary, brain state-space) assume the single-lander layout and are not meaningful for a dual run.

**Pipeline, checkpoints and data for Dual land** are covered by the Pipeline, Checkpoints and Data tabs (sections 5c-5e): there is a dual expert (one per lander), a dual cloning / DAgger pipeline, a one-click dual copy of any single-lander checkpoint, and the Data tab reads dual files (40 inputs / 10 outputs) by themselves.

---------------------------------------------------------------------------------------------------

## 5b-2. Dual land with TWO networks (start profile "pair")

The profile **dual** steers both landers with ONE network (40 inputs, 10 outputs). If that is too much for one network, use **pair**: the same environment (two landers, two pads, shared wind, landing-together reward) but **each lander has its own neural network** with its own weights, its own critic and its own optimiser, both learning at the same time.

- **What each network sees** (40 numbers): its OWN lander's 18 observations first, then its partner's 18, then height gap (own minus partner), fall-speed gap, own down, partner down. **What it writes:** its own 5 actions. Network B sees the world from lander B's side (own first), so the two are exact mirror images of each other.
- **What each network is rewarded for:** its own lander's normal reward (landing score, speed and position shaping, crash penalties...) plus the shared landing-together terms (height-gap shaping and the bonus), which both earn in full. A lander that crashes is only blamed on its own network; the partner still gets credit for a good landing.
- **How it trains:** two ordinary Stable-Baselines3 PPO models run in two threads on views of one shared environment; the environment steps once both have chosen their actions. Same run folder, curriculum, staged rules and live stream as any run (`logs\<run>\`), the same episodes.csv (joint success, `sync_dt_s`), about 400 steps/s with 2 environments on this machine.
- **Files:** a pair is two checkpoints, `<name>_A.zip` and `<name>_B.zip`; every tool takes the base name `<name>` (the Checkpoints tab shows the pair as one row of kind *two networks*). A run saves `<run>_A/_B` at the end and `<run>_level<N>_A/_B` at every level change.
- **Starting one:** *Checkpoints* tab > select a single-lander checkpoint > **Make a pair of networks** (or `python -c "from src.training.pair import make_pair; make_pair('sx_imitation_quiet', 'pair_quiet')"`). Each network starts as a copy of the single-lander network (its weights on the 18 own inputs, zero on the 22 partner inputs), so both already land their own lander and both land together at about the same time (20/20 joint successes at level 0, median gap 0.05 s); PPO then teaches each network to use the partner inputs. In Parameters set **Start profile = pair**, *Continue from checkpoint* = the pair's base name (or simply a single-lander checkpoint: both networks start as copies of it), learning rate 1e-4 to 3e-4, entropy 0.
- **Watching:** the Live view is the same as for dual (two landers, two pads). The Brain tab, the mini network and the telemetry follow the **focused** lander (key B / the Focus button) and show THAT lander's own network (40 inputs: own lander, then partner, then the gaps). The Parameters of the **Dual land** group (spacing, height, jitter, tolerance, bonus, shaping) apply to both dual profiles.
- **Evaluate:** `python -m src.sim.evaluate <name>` (or the Checkpoints tab) flies both networks deterministically and reports joint successes and the landing-time gap.
- **Giving both networks an idea of what to do first:** the Pipeline tab's *Dual land, TWO networks* profile (section 5c): the scripted expert flies pairs, each network is cloned on its OWN lander's recorded view and actions, then improved by DAgger rounds, and the best round is saved as the pair `<name>_final`. The shortcut (no pipeline) is *Make a pair of networks* from a single-lander checkpoint, above.

## 5c. Pipeline tab: expert -> clone -> DAgger -> evaluate, from the browser

The Pipeline tab runs the whole section-5 procedure for you, for **one lander or Dual land**, and shows it live. It starts `python -m src.sim.pipeline` in the background (so you can close the page; use Stop to cancel) and follows its `logs\pipeline\<name>\status.json`.

1. **Profile**: *single lander*; *Dual land, ONE network* (two experts, one per lander, the 512 x 512 network flying both); or *Dual land, TWO networks* (the same two experts, but each lander's own 40-input view and 5 actions go to its own data set `<name>_expert_A` / `_B`, and each network is cloned and DAgger-retrained on its own data; every checkpoint stage is a pair: `<name>_clone`, `<name>_dagger_N`, `<name>_final` = files `..._A` and `..._B`).
2. **Pipeline name**: everything is named after it: checkpoints `<name>_init`, `<name>_clone`, `<name>_dagger_1..N`, **`<name>_final`**; data `<name>_expert`, `<name>_dagger_1..N`. A name that already has outputs is refused unless you tick *Replace old outputs* (it then deletes `<name>_*` checkpoints and data after a confirmation).
3. **Start from**: a fresh random network, or an existing checkpoint of the same kind (for two networks: a pair, by its base name). For both Dual land profiles you can instead pick **copies of a single-lander checkpoint** (one two-copy dual network, or two separate networks): that network already flies both landers like the single one, so you can untick *clone* and go straight to DAgger (cloning on little data can only make it worse).
4. **Steps** (tick the ones to run) and their settings: demonstrations (episodes kept, action noise, level mix), cloning (epochs, learning rate, batch, exploration log-std, extra weight on the last 400 m), DAgger (rounds, episodes per round, epochs, learning rate, noise, levels, log-std, weight), evaluation (episodes per level, levels). The defaults are the ones that produced `sx_imitation`.
5. **Progress**: a badge per step with a one-line detail, a table of DAgger rounds (success per level, the landing-time gap for dual, mean, the best round marked), the log, and at the end the **final checkpoint** with *Continue with PPO...* (opens Parameters with the model and the matching start profile filled in) and *Show in Checkpoints*. **Earlier pipelines** lists every pipeline that has a status file; click one to look at it again.

The final checkpoint is the **best round by mean success over the evaluated levels** (ties go to the later round). Rough times: demonstrations about 1 min per 400 landings, cloning 2-5 min, 10-15 min per DAgger round (the dual pipeline is a little slower), evaluation about 1 min.

Command line equivalents:
```powershell
python -m src.sim.pipeline --name myrun                                              # single lander, all four steps
python -m src.sim.pipeline --name mydual --dual --from-single sx_imitation_quiet --rounds 4 --episodes 300
python -m src.sim.pipeline --name mypair --pair --from-single sx_imitation_quiet --rounds 4 --episodes 300      # two networks, each cloned on its own lander
python -m src.sim.pipeline --name mydual --dual --from-single sx_imitation_quiet --steps dagger,eval   # needs mydual_expert data
python -m src.sim.expert_data --dual --episodes 400 --out expert_dual --make-init dual_init --from-single sx_imitation_quiet   # just the dual demonstrations
python -m src.sim.evaluate mydual_final --episodes 30 --levels 0,1,2                  # works on single and dual checkpoints
```
In a dual episode the expert labels a lander that is already down as "engine off", so the network learns to cut the engine after landing. Only pairs that both landed softly within the tolerance are kept as demonstrations.

## 5d. Checkpoints tab

Every saved network, one row each: **kind** (single = 18 inputs / 5 outputs, dual land = 40 / 10), the layer sizes, the **exploration noise** (the std PPO adds to every action; 0.2 by default), the **last evaluation** (mean success and per-level soft landings; for dual also the median landing-time gap), where it came from (cloned from..., PPO run...) and its age. Click a row, then use the buttons:
- **Make a Dual land copy** (single checkpoints only): two copies side by side, 512 x 512, zero weight on the four extra inputs. It lands both landers exactly like the single one; PPO then learns to use the extra inputs to land together.
- **Make a quiet copy**: the same weights with the exploration std set to the number you give (0.08 is a good value for a cloned policy; the default 0.2 knocks a clean lander off its flare).
- **Evaluate**: flies it deterministically for the given episodes and levels in the background (about 10 s per 30 episodes per level) and stores the result next to the checkpoint (`models\configs\<name>.eval.json`), so the table remembers it.
- **Continue with PPO...**: Parameters tab with *Continue from checkpoint*, the sweep off and the **start profile set to match the kind** (a dual checkpoint only works with the dual profile and the other way round).
- **Use in the Pipeline...**: pre-fills the Pipeline tab (a dual checkpoint as the start network, a single one as the "two copies of" source).
- **Move to trash**: moves the checkpoint and its settings / evaluation files to `models\_trash\<name>_<time>\` (nothing is deleted for good; a checkpoint that the running trainer is writing is refused).
New names never overwrite an existing checkpoint.

## 5e. Data tab with dual data

The Data tab (export, import, clone) now works for both kinds. Trajectory files carry their own sizes: a dual run logs `o0..o39` and `a0..a9`, a single-lander one `o0..o17` and `a0..a4`; the readers detect that from the file, the source list marks dual data (DUAL), and the model lists mark dual models. **Cloning checks the pair**: a dual model with single data (or the other way round) is refused with an explanation, and files of both kinds cannot be mixed in one cloning job. The clone form has two extra settings: *Exploration log-std after cloning* (empty = unchanged) and *Weight of the last 400 m* (for dual data the height of both landers counts). *Continue training this model...* sets the matching start profile.
Dual episodes log both landers in one row per step; the logged reward is the combined (scaled) reward.

---------------------------------------------------------------------------------------------------

## 6. Watching a run (Live tab and the stage)

In Advanced the app has eight tabs: Live, **Startup** (0), Brain, Parameters, **Pipeline** (5c), **Checkpoints** (5d), Data (5e / 8) and Tests (Simple shows Live, Startup, Settings and Models). The Live tab is a 16:9 picture (1920x1080) with the old header text in the HUD line at its top left. Everything else (run selector, speed, pause, layout, camera buttons, chips, training controls, tabs) is on the page below the picture: scroll down.

**Hidden controls on the picture**
- Click = pause / resume (a click on a lander selects it); hold the left button = fast-forward (2x, then 4x after 1.5 s); drag = pan; **Ctrl + scroll** or pinch = zoom (plain scroll scrolls the page); double-click = auto camera.
- Keys: **Space / K** pause, **F** full screen, **H** hide / show the panels, **C** auto camera, **V** follow cam, **- / +** zoom (no limit), **L** wide / tall layout, **1 / 2 / 4** speed.

**Scale and cameras**
- The booster is drawn at **true scale** (41 m tall): about 9 px high from 3 km, ~50 px at 500 m, ~90 px at 300 m, then it fills the picture. A ring marks it while it is too small to see; the ring is not part of the vehicle.
- **Camera: auto** follows the selected lander and zooms with its altitude (smoothly, in log scale). **Follow cam** (button or V) locks onto the selected lander at full scale (it fills about half the picture height) so you can watch its corrections; the wheel (Ctrl + scroll) or - / + sets its zoom, dragging leaves it. **Free**: drag and zoom yourself. Zoom-out is unlimited (down to 1 pixel = a million km); the altitude ladder and grid adapt.

**Admin view** (both modes; the **Admin view** button in the Live bar under the picture, or key **A**): every telemetry, network and graph panel shows **all landers at once**, each in its own colour (the colour of its swatch; the second lander of a Dual land episode has its own hue). It is deliberately dense and may be unreadable: it is for spotting which landers behave differently.
- *Telemetry*: every row has one coloured value per lander (in swatch order; a Dual land slot contributes two).
- *Neural network panels* (the mini network and the Brain tab): the activations of every lander's network are laid over each other (the network of each lander of a pair is its own).
- *Canvas panels* (nadir cam, phase portrait and scope, attitude plane, flight display, ghost path, decision boundary, critic, brain state-space, reward waterfall, energy budget): each is drawn once per lander and the passes are overlaid. The first lander keeps its own colours; the others are added as coloured differences. The fuel profile, flight recorder, landing scatter and learning heat-map already show every lander and stay as they are.
- To keep the page responsive the panels refresh in turns (a group per animation frame), so each one updates a few times per second. Switch the view off with the same button or key; panels go back to following the selected lander.

**The scene:** the ground is the sea (animated swell) and every landing pad is a drone ship seen from the side: the deck is the pad at height 0 (the physics is unchanged), with yellow trim, hull with rust band and blue thruster pipes, containers, bridge, mast and radome at the right end, a yellow-and-white landing circle whose width is the 10 m success zone, and a white X at the centre. Dual land shows two ships, PAD A and PAD B. The water line is 1.6 m below the deck; a touchdown off the ship is still judged at height 0.

**Seven abstract panels** (Graphs row of chips, all off by default; each is draggable, resizable and unloadable like the other graphs, and follows the lander selected with the number buttons):
- **Vitals rings**: six glowing activity rings (fuel, altitude, speed against the safe envelope, throttle, tilt, RCS gas) with a radar sweep and the altitude in the middle.
- **Actuator river**: the five commands (throttle, gimbal, fins, RCS A, RCS B) as a flowing, stacked streamgraph over the last 12 s.
- **Action galaxy**: every step of the episode is a star on a slowly turning spiral (the start at the core); size is the throttle, colour how hard it steered; RCS firings are white sparks.
- **Airflow field**: tracers streaming past the stage the way the air does (against its velocity, shifted by the wind), bending round the body, hotter and brighter with dynamic pressure. The stage is a picture, `src/app/models/airflow_stage.png` (a transparent PNG, nose up, engine at the bottom; replace it to change the model; `AIR_ENGINE` in index.html is where the flame starts, as a fraction of its height), drawn large and tilted with the lander, with a flame under it when the engine fires.
- **Descent tunnel**: a ring every 100 m of height rushes past as you fall; being off the pad moves the vanishing point, the tilt turns the tunnel, the pad glows green at the end.
- **Sync braid** (pair only): the two landers' paths twist round each other like a double helix; the further apart their heights, the wider the braid and the redder the rungs.
- **Telemetry equalizer**: seventeen channels as bouncing LED bars with peak-hold marks and a reflection.
They are described in the `XP` list in index.html (one entry each, plus a draw function), so adding another is easy. They are drawn about 15 times a second like the other graphs and are not part of the Admin view.

**The partner cam (pair / Dual land only):** a small square camera in its OWN panel next to the landing cam: drag it by its header, resize it with the corner grip, collapse it with the arrow, and unload it completely with the **Partner cam** chip (Panels row; then the panel is hidden and nothing is drawn for it, which saves the performance; the Landing cam chip does the same for the main camera). **Layout > reset panels** puts it back. It shows the same picture from the OTHER lander of the pair (so it follows the Focus A/B switch), cut to the middle square, with its own broadcast glitches and grime, and a re-arranged display: the lander's number (its pad, as on its fuselage) and letter, the distance to its own ship, speed and altitude rings in the bottom corners, flight time, tilt, spin and g-force in the middle, an engine light, and a frame in the lander's colour. It redraws 8 times a second (`CAM2_FPS` in index.html). Like the other panels it is always there while its chip is on (so you can place it), and shows a red PAIR ONLY badge until a pair episode is on screen.

**The fire** has a white-hot centre that fades through pale yellow and yellow to orange at the edges, in the main view (a stack of layers plus a white-hot ball at the nozzle exit) and in the landing cam (hot-core sprites along the plume and a brighter bell glow).

**Lens flares:** bright lights get a cinematic anamorphic look, a long, soft (blurred) horizontal streak with a white core and a soft halo, red-orange for the main engine and cool blue for the RCS thrusters, placed at the nozzle exit, the end of the bell where the flame starts: the main engine (longer and brighter with the throttle) and every firing RCS thruster, in the main view, and the main engine only in the landing cam (no RCS flares there; the streak is part of the picture, so it also shakes and blurs with it). They follow the *Flames* chip. The strength is `FLARE_STRENGTH` (0.55; 1 = full) next to `anam()` and `flares()` in index.html.

**Effects** (chips under the picture; remembered): airborne / ignition / landing points, end notices, rings, trails, **flames** (main plume with sheath, core, shock diamonds and a ground splash; RCS jets), **smoke** (billowing cloud sprites: exhaust smoke is warm and rises; when the plume reaches the deck a fast sheet of steam and spray shoots out along the surface, slows down and rolls up into a tall plume, like a real landing; it is made of very few, very big clouds to stay light: `SMOKE_N` (how many, 0.085) and `SMOKE_SIZE` (how big, 3.0) in index.html, a soft limit of 360 at once; clouds entirely off screen are not drawn. Smoke never vanishes abruptly: a cloud that has to go, because there are too many or because a new episode started, spreads and fades over 1.4 s (`DISSIPATE_S`)), ID tags, grid. The stage is drawn after a real Falcon 9 first stage (white tank section with soot toward the engines, black interstage; the top of the white tank, just under the interstage, is painted in the lander's colour and is what tells the landers apart. In Dual land both partners share one colour; the band carries a **1** or **2** for the pad that lander is to land on (1 = pad A, 2 = pad B), carbon legs with a bright telescoping lower strut, octaweb with bells; the lander's colour is also a faint outline). Fins are drawn deployed as lattice panels at the top of the stage and swing with the fin command (drawn x1.6); gimbal is drawn x4.

**Panels**: network (with fine-scale input channels), terminal (the flight log of the selected lander), telemetry, landing cam (a camera fixed to the upper part of the stage, 30 m above the engine bells and 1.5 m off the skin, looking down the length of the stage: a real 3-D view of the sea and the drone ships, ray-traced from the lander's position and attitude, so the recorded **tilt** swings the world across the picture (the horizon appears when it tilts far) and the **spin** smears it sideways. The hull, the two landing legs and the two grid fins are one fixed overlay picture. **Every transparent PNG in `src/app/cam_hulls/` is one such picture** (four now: `hull_1.png` to `hull_4.png`): each episode picks one for the landing cam and always a *different* one for the partner cam, and the view slides a little up or down so that the engine's flame sits exactly at the top of whichever hull is shown (the height of the hull's top is measured automatically). Drop in more 16:9 transparent PNGs, with the hull along the bottom and the legs meeting at its top, to get more variety (reload the page). `src/app/cam_hull.png` is only the fallback while they load. A hull ray-traced from the stage's real geometry is kept as `cam_hull_generated.png` (made by `python src/app/make_cam_hull.py`; copy it over `cam_hull.png` to use it). Everything in the overlay is part of the picture and never moves. The engine plume, smoke, spray and RCS puffs are the viewer's own particles seen from the camera, and the engine lights the hull when it fires. **The feed misbehaves now and then, like a compressed live video**: every several seconds (sooner when the stage shakes hard) the picture either FREEZES for a moment (0.25 to 0.9 s), or part of it BREAKS UP: a patch of blocks stays stuck on an old frame while the rest keeps moving and others turn into coarse blocks, then it heals block by block and comes back with one soft, blocky frame. Nothing else (no scanlines, noise or colour glitches). `ARTIFACTS` in index.html: 0 = clean feed, 1 = default, 2 = a bad connection. The display on top stays live. The shake is also a little stronger than before. **The lens is a little dirty**: smudges, greasy smears, water drops, dust specks and a few hairs, fixed to the camera (so they shake with it, and blur with the picture), a different pattern for every episode (it stays the same for the whole episode); when the engine fires its light scatters in the dirt and the grime glows. `GRIME` in index.html: 0 = clean, 1 = default, 2 = filthy. **The camera is shaky and unstable**: the whole frame jitters and tilts a little, and its rows ripple sideways in a wave that runs down the picture, only a slight shake while the main engine is off (a little more in thick, fast air), much stronger while it fires, and a small bump for every RCS puff (the display stays steady). **The whole picture (scene, hull, legs, plume, smoke) is blurred together**, with a streak away from the vanishing point that grows with the fall speed and a slight sideways smear from the spin; the on-screen display stays sharp: speed, altitude, flight time, tilt and spin, g-force and the engine cluster, and the top line gives the distance to the ship, the coast miss and the time to impact. It is rendered small (192 x 108, 12 pictures a second (`CAM_FPS` in index.html)) so it stays light on the page), phase portrait, and twelve optional graph panels (Graphs chips). Panels drag by their header and scale by their corner grip; **Layout > reset panels** restores them. A panel with no data keeps its last picture and shows a red "PENDING DATA" badge.

**Brain tab**: the selected lander's network (18 inputs, 256 + 256 neurons, 5 outputs: main engine, gimbal, fins, RCS A, RCS B) lit up frame by frame, what it senses, what it decided, and its own estimate of how well it is doing. Needs a run started after neural-data logging was added (older runs say so).

**Descent progress graph** (under Training progress): per 25k-step bin: in safe-speed envelope, braking headroom, touchdown softness, landing accuracy, merit. They move before the first landing does.

---------------------------------------------------------------------------------------------------

## 6b. Showcase interface (for a public display, almost no load on the laptop)

The third interface next to Simple and Advanced (the Showcase Recorder button beside it makes its recordings). It loops through **saved landings** forever and shows the data behind each one for a reader who wants numbers. It runs no simulator, no neural network and no training, and polls nothing: the page only draws recorded frames (the same lander, drone ship, flares, smoke and landing cam as the live view) at up to 30 pictures a second.

**1. Record the library once, from the browser.** Press **Showcase Recorder** in the *Interface* switch (or open `http://127.0.0.1:8767/?recorder=1`). The screen shows the saved library (a table of its landings and when it was recorded) and a **Record a new library** card:
- **Successes per wind level** (1-4, default 2) and **Failures to include** (0-3, default 3). The network is fixed to `rt1_final`, the one the evaluation table in `paper_assets\retrain\` was made with.
- **Record landings** starts it; the bar and the line under it say "flying the landings... 4 of 11" with the seconds so far, and the last lines of its log are shown. It takes about 12-15 s. **Cancel** stops it.
- **Refresh the numbers only** rewrites just the numbers beside the landings (a few seconds), for when you changed the results files in `paper_assets\retrain\`.
- The saved library is replaced only when the new one is complete (it is built in `replays.tmp\` first and moved over `replays\` at the end), so a failed or cancelled recording leaves the old one as it was; a failure shows the exit code and the log. Only one recording runs at a time. You can leave the page: it picks the state up again when you come back. Pressing **Open the Showcase** shows the result; a Showcase that was already open needs a refresh.
- The screen runs nothing else: no live view, no polling except its own progress while a recording runs.

The button runs exactly this command, which still works from a terminal (`--per-level`, `--failures`, `--facts-only`, `--checkpoint`, `--out` are its options):

```powershell
python tools\record_replays.py
```

It flies `rt1_final` again with the paper's seeds and the same deterministic policy (rules stage 2) and stops if a recorded touchdown speed differs from the evaluation table. It writes into `replays\` (about 2 MB, committed to the repository so a fresh clone can show the showcase without recording):
- `ep_NN_L<level>_s<seed>.json`, one per landing: 11 by default: the paper's featured one (level 1, seed 122, touchdown 2.09 m/s), two typical successes at every wind level and three typical failures (levels 1-3, never two in a row, never first). Options: `--per-level N`, `--failures 0..3`, `--checkpoint <name>`, `--out <folder>`.
- `index.json`: per landing the outcome and its measured numbers (start state, ignition altitude and burn time, max-Q, peak load, touchdown speed split into vertical and sideways, tilt, pad offset, propellant, merit, wind, and its rank among the 300 evaluated seeds of its level).
- `facts.json`: the numbers for the slides, read from `paper_assets\retrain\summary.csv` and `attempts.md`, `src\utils\config.py` and the checkpoint (success rates with 95 % Wilson intervals for the retrained network, the scripted expert, the earlier network and the pair, failure reasons, the DAgger validation curve, the level table, rules, checkpoint SHA-256 and parameter count). `python tools\record_replays.py --facts-only` rewrites only this file after those source files change.

**2. Start it.** Start the app as usual (`python src\app\server.py --port 8767`, or `run_app.bat`; restart a server that was started before this feature existed) and either press **Showcase** in the *Interface* switch under the picture, or open

```
http://127.0.0.1:8767/?showcase=1
```

(bookmark it for a stand). Press **F** for full screen. Nothing needs to be clicked afterwards. To go back, either use the switch at the top of the right-hand column, or **scroll down**: the usual page (Training card, the *Interface* switch, the Live / Startup / Settings / Models tabs and everything in them) sits under the picture exactly as in the normal view. **Simple** or **Advanced** in either place returns to the normal page, **Showcase Recorder** opens the recorder.

**What is on the screen.** Left: the picture with its usual panels (telemetry, phase portrait, flight terminal, landing cam; they follow your Chips settings) and a caption. Right, a column:
- **This landing** (always): wind (scale, this episode's mean, gusts), start (height, speed, entry angle, offset), ignition time and altitude, burn length, max-Q and peak load, touchdown speed (vertical and sideways), tilt, distance from the pad centre, propellant used and left, merit, share of the descent inside the safe-speed envelope, peak braking demand; after touchdown also its rank among the 300 evaluated landings of that level and a tick or cross for each of the five success rules.
- **Slides** that turn every 18 s (a click on a tab holds the slide for a minute): **Results** (bar chart and table, 95 % intervals, four controllers), **Touchdown** (median and mean speed, pad offset, tilt, propellant per level), **Failures** (what broke, per level and controller), **Training** (validation success after cloning and after each DAgger round), **Task** (the four wind levels and the success rules), **Method** (simulator, network, teacher, evaluation protocol, checkpoint hash) and **Limits** (what is not shown or not proven). On a screen shorter than about 700 px a long slide is shrunk to fit.

**Timing.** Each landing plays at normal speed, stays on screen for 3.5 s, then a 1.8 s "up next" caption, then the next one; after the last it starts again. Every new landing resets the camera and the speed, and a picture a visitor paused resumes after 30 s. The Network panel (it needs recorded activations) and the partner cam (pairs only) are hidden. The usual page under the picture does nothing while you are looking at the picture (no requests, no live updates); it wakes when you scroll down to it and goes quiet again when you scroll back up, and the picture is not drawn while it is scrolled out of view (the replay carries on where it stopped). Below 900 px of window width the data column is left out.

**Why it is cheap.** The drawing stops when the tab is hidden or minimised; frames are capped at 30 a second; only the current and the next landing are in memory (about 0.2 MB each as text); every list that grows during a long run (smoke particles, terminal lines, landing-cam textures) already has a cap; the data column is rebuilt only when a landing or a slide changes. The server does almost nothing while it plays: one small request per landing (every 20-30 s). Measured on this laptop (20 logical cores) with the first version of this mode (picture only): the server used 0 % CPU (about 495 MB of memory, mostly the Python libraries it loads) and the page's drawing code about 9 % of one core (roughly 0.5 % of the whole machine) at about 30 pictures a second. The data column adds a little text work per landing and per slide and was not measured separately; the browser's own rendering and graphics-card work is not included in any of these figures. Stop any training and leave other tabs closed for the lightest result.

**Troubleshooting.** "No saved landings yet" on the picture: run step 1; the page checks again every 5 s. After re-recording while the page is open, refresh it. If a slide says there is no results file, run `python tools\record_replays.py --facts-only`. The server must be a copy started after this feature was added (section 9, "An option is missing"), otherwise `/api/replay/...` does not exist.

---------------------------------------------------------------------------------------------------

## 7. Reading a run: what good and bad look like

Look at `logs\<run>\episodes.csv` (or the Live charts). Useful columns: `success`, `end` (rest / crash / tilt / stall), `speed_mps`, `dist_m`, `fuel_used`, `ignitions`, `level`, `rules_stage`.

- **Success** = at rest within 10 m of the pad centre, at most 3 m/s, tilt at most 0.15 rad. "Rest" without success = survived (under 5 m/s, tilt under 0.2) but not soft or not on the pad.
- A run that is **learning to land** shows the crash share falling and touchdown speed trending toward 2-3 m/s with ignitions near 1.
- A run that is **unlearning** shows touchdown speed rising (13 -> 20+ m/s), ignitions per episode jumping to double digits (engine chatter) and no soft landings. Typical causes: exploration noise too high for a cloned policy (use `sx_imitation_quiet`), learning rate too high, or the level moved up before the previous one was solved.
- Check `levels.csv`: a row with reason `max_steps` means the level advanced on the step limit, not because the gate passed. If the success rate in that row is low, the agent is now being trained on a harder task it cannot solve. Raise *Max steps per level* or use *Stop when this level's gate passes* (-1 = never) to hold the level.

Fuel: the budget is 4,500 kg of landing propellant; the expert needs a median of about 2.1 t (level 0) to 2.4 t (level 3) in the flights and 2.9-3.3 t in the calibration with its margin. `python -m src.sim.calibrate --n 60` recomputes it after any vehicle change (writes `models\configs\calibration_spacexperiment.json`; put the recommended budget into `FUEL_BUDGET` or the Parameters tab).

---------------------------------------------------------------------------------------------------

## 8. Data tab: export, import, train further (see also 5e for dual data)

- **Logged:** every run writes step-level experience (`o0..o17` observation, `a0..a4` actions, scaled reward, done flag) to `data\trajectories\<run>\`; every Nth episode (default 10) plus the best and worst 5% are kept; total capped at the data budget in Parameters (default a few GB; the oldest chunks go first).
- **Export** trajectories, episode summaries or a model's weights as `.csv`; **import** weights or trajectories from a `.csv` (a new model, or overwrite one after a backup copy).
- **Train further:** *Clone from data* (behaviour cloning + value warm-up, filters by merit / return / soft landings; the source model is never changed) or *Continue training with PPO* (Parameters > Run > Continue from checkpoint).
- Why not replay old data through PPO: PPO needs the action probabilities of the policy that produced the steps, so stale data would silently train on the wrong thing. Cloning only needs observation to action pairs.
- Files written before the 18-input observation have 14 observation columns and cannot be used with the current network. Single-lander (18 / 5) and Dual land (40 / 10) files cannot be cloned into the other kind's network.

---------------------------------------------------------------------------------------------------

## 9. If something looks wrong

| Symptom | Fix |
|---|---|
| "address already in use" when starting the app | An old server is still on 8767. Use the leftovers command (with `server.py`), or `.\run_app.ps1 -Port 8768`. |
| Page says *server unreachable* | The server window was closed. Restart it. Training keeps running without it. |
| An option is missing (for example only one start profile) or a new tab does nothing | The server is an old copy: Python changes (the Parameters schema, the tabs' endpoints) only load when the server starts. Stop it and start `.\run_app.ps1` again, then refresh the page. |
| Start says "already running" | A run is active (possibly started outside the app). Press Stop. |
| Live scene is empty after Start | Wait 10-30 s; if still empty read the Training panel's log lines or `logs\<run>.out`. |
| The page is blank or old after changing files | Hard-refresh the tab. The server sends the page fresh on every request but the browser may cache scripts it keeps. Python changes (physics, env, expert) need the server and any trainer restarted. |
| Clone lands in evaluation but crashes in PPO | Exploration noise: continue from the quiet checkpoint (section 5). |
| DAgger or cloning results change between identical runs | Two copies were running at once, writing the same `dagger_N` folders. Kill all (leftovers), delete `data\trajectories\dagger_*`, run one. |
| Batch size error | Batch size must divide n_steps x environments. |
| Training is slow | CPU-bound; fewer environments help the page more than the trainer. Keep device on `cpu`. |
| Smoke or flames slow the page down | Switch off **Smoke** (chip) or lower the speed; the particle count is capped at 2,400. |

---------------------------------------------------------------------------------------------------

## 10. Reference

- Tests: `.\run_tests.ps1` or `python -m pytest tests -q` (86 tests). Parts of the old suites that touch the trainer start real training processes and can be flaky under heavy CPU load: rerun a single failure in isolation before suspecting a bug.
- Sections of the Rocketeer runbook about 1 m landers, 50 steps per second, the 11-input 64-unit network, the 7.45 m/s^2 brake limit or "no atmosphere" do not apply here. In SpaceXperiment: 20 decisions per second, episodes of about 500-900 steps, the 18-input 256 x 256 network, ~21.7 m/s^2 net braking at full thrust (T/W 3.2 full, 1.3 at the 40% throttle floor), terminal velocity 184 m/s at sea level, drag and wind on.
- Simulator cost: about 0.2 ms per step per environment (the Box2D original needed about 1 ms).
