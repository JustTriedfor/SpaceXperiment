# SpaceXperiment

A separate version of the project (copied from Rocketeer, which lives one folder up and is untouched): a **real-scale Falcon 9 landing burn** on our **own 2-D physics engine** (no Box2D). The agent starts 3 km above the pad at up to ~300 m/s, coming in from the side like a re-entry, and must brake, steer and land.

Run it: `.\run_app.ps1` (port **8767**, so it can run next to the original on 8765). Tests: `python -m pytest -q`.

## What is simulated (src/sim)
Frame: x right (pad centre x = 0), y up (ground y = 0); the body axis points from the engines to the nose; the booster is integrated about its centre of gravity (CG). 20 decisions per second, 5 physics substeps (0.01 s) each. g = 9.81 m/s^2.

| Part | Model |
|---|---|
| Mass | 22,200 kg dry + landing propellant (budget 4,500 kg) + 90 kg RCS nitrogen; propellant burns, so mass, inertia and weight fall during the burn |
| Main engine | one Merlin 1D: 845 kN sea level, 914 kN vacuum (altitude-dependent), Isp 311 s vac -> 299.7 kg/s at full throttle; **throttle 40-100% or off** (cannot hover: T/W 1.3 at minimum), spool lag 0.25 s, up to 3 ignitions |
| Gimbal | centre engine thrust vector +-5 deg, 20 deg/s, lag 0.1 s |
| Grid fins | lift = dynamic pressure along the axis x 6 m^2 x 2.0/rad x deflection (+-25 deg), arm 24 m above the CG |
| RCS | **two cold-gas systems**, A (21 m above the CG) and B (11 m below it), 5 kN each, 90 kg of gas shared; together they translate, opposed they rotate |
| Air | exponential density (scale height 8.4 km); axial drag (Cd 1.2 engine-first, 0.55 nose-first, 10.5 m^2); cross-flow drag on three body slices (stable engine-first and damped) |
| Wind | per-episode mean + slow gust, stronger aloft, by curriculum level (0 / 5 / 10 / 15 m/s) |
| Legs | 18 m span, feet 1 m below the engines; touchdown ends the episode (rest if vertical speed <= 5 m/s, sideways <= 3 m/s, tilt <= 0.2 rad, else crash) |

**Sources and honesty.** The mass, size, Merlin thrust / Isp / throttle floor and the ~1-2 m/s best real touchdowns are approximate public figures. Everything marked "estimate" in `src/sim/vehicle.py` (grid-fin area and lift slope, RCS thrust and gas, cross-flow drag, centre of gravity, actuator lags) is a modelling choice, not SpaceX data. Recalibrate them there (they can be overridden through `ROCKETEER_OVERRIDES["vehicle"]`).

## Calibration (python -m src.sim.calibrate)
- Engine table at the landing mass (~26.8 t): T/W 3.2 at full / 1.3 at minimum throttle, net braking 21.7 m/s^2 at full throttle, terminal velocity 184 m/s at sea level.
- Fuel: a scripted expert (suicide-burn controller, `src/sim/expert.py`) flies every level with plenty of propellant; surviving landings need a **median of about 2.9 t (level 1) to 3.3 t (level 3)**. Budget = 1.5 x the median of those = **4,500 kg**. Results: `models/configs/calibration_spacexperiment.json`.
- The expert (`python -m src.sim.eval_expert`) lands **softly and inside the 10 m circle in 100% of level-0 starts, ~75% at level 1, ~36% at level 2 and ~13% at level 3** (150 fresh seeds each). What it does: coast engine-first, one latched burn that follows a braking profile (throttle from the deceleration the remaining height asks for), lateral zero-effort-miss guidance with a look-ahead for the tilt-loop lag (thrust tilt capped by height so it touches down upright), fins at high dynamic pressure / gimbal under thrust / RCS as backup for attitude, and a cut-and-burn-again recovery if it ever stops above the ground and starts to climb (the 40% throttle floor cannot hover). Levels 2-3 start with 60-75 m/s of sideways speed at the burn start, more than the sideways authority of a thrust-tilted stage can remove in the ~8 s of the burn.

## The task
- **Spawn** (`src/sim/spawn.py`): 3,000 m above the pad, speed 250-320 m/s x the level fraction (35%, 60%, 80%, 100%), flight-path angle 72-85 deg below the horizontal at level 0 down to 42-72 deg at level 3, engine-first with a small angle of attack. The start position is found by **simulating a nominal landing** (coast with drag, straight braking burn, and a constant-rate sideways stop, `physics.powered_descent`) and shifting the start so that this path ends within the level's miss allowance (50 m at level 0, 350 m at level 3) of the pad centre. (An earlier version aimed the *passive* fall at the pad; the burn lasts ~9 s longer than the fall, so every normal landing then overshot by 100-300 m.)
- **Observation** (18): x, height, vx, vy, tilt, spin, propellant, thrust level, gimbal, fin angle, RCS gas, engine on, ignitions used, dynamic pressure, then x, height, vx, vy again on a square-root scale (fine resolution for the last metres). **Action** (5, each -1..1): throttle (<= -0.5 off, else 40-100%), gimbal, fin, RCS A, RCS B.
- **Rules** (staged as before): tilt limit 75 deg; stall (stage 1: below 6 m/s above 100 m); descent clock (stage 1: at least 1.5 x the fastest possible soft landing from the spawn + 8 s, floor 30 s); ignition limit 3 (stage 2); fuel = real propellant. Success = within 10 m, <= 3 m/s, tilt <= 0.15 rad.
- **Reward**: the same design as REWARD.md rescaled to real units (see section 13 there). All constants are at the top of `src/training/reward.py` and in the Parameters tab.
- **Network**: 256 x 256 for the actor and for the critic (`NET_ARCH`), PPO batch 512, entropy 0.005. Measured on this machine (8 environments, CPU): 833 steps/s with 256 x 256 against 1,387 with 64 x 64 - the bigger network costs about 40% of the throughput; batch 1024 gives 957 steps/s, and 12 environments 1,028.

## What the viewer shows
The stage follows the selected booster and zooms with its altitude (a 3 km descent down to the last metres). The booster is drawn to scale (41 m, 3.66 m wide; the glyph is magnified when zoomed far out): engine plume along the **gimballed** nozzle (gimbal drawn x4), **grid fins that rotate** with the command and show their lift, **two RCS systems** that puff (A cyan at the top, B violet near the engines), deployed legs, and a re-entry glow at the engine end while the dynamic pressure is high. Telemetry, nadir cam, ghost path, flight display, etc. all use the real numbers (kN, kPa, Mach, kg, drag).

## Known limits
2-D only; flat ground; no Earth curvature or rotation; no fuel slosh; constant CG; no engine start-up transient beyond a first-order lag; no aero heating model (the glow is cosmetic); grid fins modelled as one lifting pair; no aerodynamic stall of the fins; touchdown is a pass/fail check, not a contact simulation (no legs bouncing or tipping over). The learning difficulty at full speed with wind is unproven: expect to use the curriculum levels.

## Starting from an expert (imitation)
`python -m src.sim.expert_data --episodes 400 --out expert_sx --noise 0.12 --levels 0,0,1,1,2 --make-init sx_init` flies the scripted expert and logs 400 good landings (noise-augmented: the executed action is jittered, the logged one is clean); `python src/training/finetune.py --checkpoint sx_init --data expert_sx --out sx_expert_clone --epochs 40 --lr 0.001 --batch 512 --log-std -1.6 --low-weight 5` clones it. Then `python -m src.sim.dagger_loop` runs six **DAgger** rounds (the cloned policy flies 160 episodes, the expert labels every state it visits, the policy is retrained on everything, low-altitude samples weighted x5), and `python src/sim/eval_policy.py <checkpoint> 100` flies a checkpoint deterministically.
- Result, **`models/checkpoints/sx_imitation.zip`** (100 fresh episodes per level): **level 0: 100/100 soft landings** (median 2.4 m/s, 2 m off-centre, 2.1 t of propellant); **level 1: 70/100**; **level 2: 45/100**. That equals or beats the expert it learned from (100% / 75% / 36%). Round by round, level-0 touchdown speed went 31 -> 9 -> 3 -> 2 m/s.
- What made the difference: (1) the expert itself (see above) - the first attempt cloned an expert that landed softly only ~30% of the time; (2) **four extra observation channels** (18 inputs now): x, height, sideways speed and fall speed again on a square-root scale. On the linear scale the last 20 m of a landing is 0.007 of the input range, and the policy never learned the flare (it stalled at ~10 m/s touchdown); with the fine channels it does.
- Start PPO from it with `init_from` (all rules on, entropy 0, learning rate 1e-4); the policy is already deterministic-good, so keep the learning rate low or the first updates will un-learn the flare.

## Dual land (start profile `dual`)
`src/sim/dual_env.py`: one environment, two landers, two pads (adjustable spacing, default 1 km), both starting together 4 km up on the same predetermined path (one start per episode given to both, small jitter), one policy steering both (40 inputs, 10 outputs, 512 x 512), episode over when both are down, reward = both landers' rewards + landing-together bonus and height-gap shaping. Two copies of the scripted expert land together with a median touchdown gap of 0.05 s at level 0. `python -m src.sim.dual_init` turns any single-lander checkpoint into a dual one, `python -m src.sim.eval_dual` evaluates it. Details, parameters and the viewer behaviour are in RUNBOOK.md section 5b.

The browser app has Pipeline and Checkpoints tabs and a dimension-aware Data tab for this (RUNBOOK.md sections 5c-5e); on the command line: `python -m src.sim.pipeline`, `python -m src.sim.evaluate`, `python -m src.sim.expert_data --dual`.

Profile `pair` is the same Dual land environment with TWO networks (one per lander, own weights / critic / reward, trained together by two PPO models on views of one shared environment, `src/training/pair.py`); see RUNBOOK.md section 5b-2.
The pipeline (`python -m src.sim.pipeline --pair`, or the Pipeline tab) also builds pairs: each network is cloned and DAgger-retrained on its own lander's data.

The app opens in a **Simple** interface (Live, Startup, Settings, Models: sliders and menus within safe limits); **Advanced** is the full one. The **Startup** tab runs a head start (copy a brain, or the expert pipeline) and then training in one go (RUNBOOK.md section 0).

## Touchdown: legs with some give, and 2.5 s to straighten up
The episode no longer ends the instant a foot touches the ground (that froze good approaches that merely touched down a little tilted or on one leg first).
- **Legs:** each foot is a spring + damper with a 0.9 m stroke and a hard stop (`LEG_*` in `src/sim/vehicle.py`: 500 kN/m, 120 kN s/m, estimates), with sliding friction on the deck. The stage settles about 0.25 m onto its legs; the viewer shows the leg shortening.
- **Engine:** it shuts down at touchdown (`TOUCHDOWN_ENGINE_CUT`), as on a real booster, and stays off while the lander settles; RCS still works.
- **Verdict:** at the first touch the impact is judged as before (vertical speed <= 5 m/s, sideways <= 3 m/s) plus a touchdown tilt of at most 0.45 rad (`CONTACT_TILT_MAX`): worse than that is a crash at once. Then the lander has `SETTLE_S` = **2.5 s** (a slider in Settings / Parameters: 0 = the old instant rule). It ends early as a **rest** when it has been upright (<= 0.15 rad) and nearly still (< 0.5 m/s, < 0.05 rad/s) for 0.6 s; when the time is up it is a rest if it is within 0.2 rad of upright and slower than 1.5 m/s, otherwise a crash. Tipping past 0.62 rad is a crash.
- **Scoring:** the impact speed is the speed of first contact; the miss distance and the tilt are measured where it ended up. Success (within 10 m, <= 3 m/s, <= 0.15 rad) therefore counts a lander that touched down tilted and then straightened up. In Dual land the landing-together gap is measured between the two first touches.
- Checkpoints made before this change still land (both the single and the pair `sx_imitation_quiet` / `pair_quiet` give the same results as before) because the engine cuts for them. Runs that are already training use the rules they started with.
