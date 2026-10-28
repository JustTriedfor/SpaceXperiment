"""Tunable run parameters exposed in the web app, with validation and CLI/override mapping."""
from __future__ import annotations

import re
import sys

from src.training import reward as R
from src.utils import config as C

# target: ("arg", name) -> train.py CLI flag, ("config", NAME), ("ppo", name), ("reward", NAME)
F = []


def _f(key, group, label, typ, default, target, lo=None, hi=None, step=None, options=None, help="", always=False):
    """always=True: the app default differs from the config default, so the value is always sent as an override."""
    F.append(dict(key=key, group=group, label=label, type=typ, default=default, target=target,
                  min=lo, max=hi, step=step, options=options, help=help, always=always))


# ---- run
_f("reward", "Run", "Reward", "select", "custom", ("arg", "--reward"), options=["custom"],
   help="The point system (see REWARD.md). SpaceXperiment has no second reward type.")
_f("seed", "Run", "Seed", "int", 0, ("arg", "--seed"), 0, 10_000)
_f("total_steps", "Run", "Total timesteps", "int", 3_000_000, ("arg", "--total-steps"), 1_000, 50_000_000, 10_000,
   help="An episode is about 500-700 steps (20 steps per second).")
_f("device", "Run", "Compute device for the network", "select", "cpu", ("config", "DEVICE"), options=["cpu", "cuda", "auto"],
   help="The simulation always runs on the CPU. Measured on this machine (with the old 64 x 64 network; the bigger 256 x 256 network gains more from a GPU) (RTX 4060 Laptop, 8 environments): the GPU is about 2.2x SLOWER "
        "than the CPU (about 1,100 vs 2,400 steps/s), because the network is tiny and every simulation step needs a CPU-GPU round trip. "
        "Leave on cpu. Never use cuda in a sweep (every agent would open its own GPU context). auto = cuda if available, else cpu.")
_f("n_envs", "Run", "Parallel environments", "int", C.N_ENVS, ("arg", "--n-envs"), 1, 64,
   help="Shared mode: workers feeding one policy. Sweep mode: number of independent agents (each ~0.5 GB RAM).")
_f("sweep", "Run", "Learning-rate sweep mode", "select", "off", ("sweep", "mode"), options=["off", "on"],
   help="off: all environments feed ONE shared policy. on: each environment gets its OWN agent; learning rate = "
        "base x growth^index. Parallel environments = number of agents; total timesteps is per agent.")
_f("lr_growth", "Run", "Sweep: LR growth per environment (x)", "float", 1.5, ("sweep", "growth"), 1.0, 4.0, .05,
   help="Agent i learns at (PPO learning rate) x growth^i. Only used when sweep mode is on.")
_f("sweep_copies", "Run", "Sweep: copies per learning rate", "int", 1, ("sweep", "copies"), 1, 16,
   help="Agents sharing one learning rate (different seeds), so they can be grouped afterwards. Parallel environments must be a multiple of this.")
_f("staged", "Run", "Staged rules", "select", "on", ("staged", "mode"), options=["on", "off"],
   help="on: the newer rules switch on as the agent learns to land. Stage 0: tilt limit, slow-approach charge, crash and landing scores. "
        "Stage 1 adds the descent clock and the stall rule. Stage 2 adds the ignition limit and re-ignition penalty. off: every rule from step 0.")
_f("start_level", "Run", "Start level", "int", 0, ("arg", "--start-level"), 0, C.N_LEVELS - 1)
_f("stop_at_level", "Run", "Stop when this level's gate passes (-1 = never)", "int", -1, ("arg", "--stop-at-level"), -1, C.N_LEVELS - 1)
_f("init_from", "Run", "Continue from checkpoint (name, empty = start fresh)", "text", "", ("arg", "--init-from"),
   help="Starts this run from an existing agent's weights (a name from the Data tab, e.g. custom_s0_app or a cloned model) instead of a random network. Shared-policy runs only (sweep off).")
_f("tag", "Run", "Run tag", "text", "_app", ("arg", "--tag"), help="Run name = <reward>_s<seed><tag>. Same name overwrites that run's logs. Left at _app, the tag follows the start profile (_dual, _pair), so those runs get their own name.")
# ---- experience data
_f("data_every", "Data", "Log every Nth episode's steps (0 = no trajectory log)", "int", C.DATA_EVERY, ("config", "DATA_EVERY"), 0, 1000, 1,
   help="Step-level experience for the Data tab (export / clone). Besides every Nth episode the best and worst 5% of recent episodes are kept.")
_f("data_budget", "Data", "Disk budget for all trajectory data (GB)", "float", C.DATA_BUDGET_GB, ("config", "DATA_BUDGET_GB"), 0.1, 100, 0.5,
   help="When all logged trajectories together exceed this, the oldest chunk files are deleted.")
# ---- start conditions
_f("start_profile", "Start conditions", "Start profile", "select", "spacex", ("config", "START_PROFILE"), options=["spacex", "dual", "pair"], always=True,
   help="spacex: the re-entry side entry, 3 km up at up to ~300 m/s on a drag-simulated path aimed near the pad, engine-first with a small angle of attack. "
        "dual (Dual land): ONE environment with TWO landers and TWO pads; both start together on the same predetermined path and must land at about the same time "
        "(see the Dual land section below). The policy is then ONE network, 40 inputs -> 10 outputs: start it from a checkpoint made with python -m src.sim.dual_init. "
        "pair (Dual land, two networks): the same environment, but each lander has its OWN network (40 -> 5, its own weights, critic and reward); start it from a single-lander "
        "checkpoint (each network starts as a copy) or from a pair (name = the base of NAME_A / NAME_B).")
_f("entry_preset", "Start conditions", "Re-entry angle", "select", C.ENTRY_PRESET, ("config", "ENTRY_PRESET"), options=["extreme", "intermediate", "SpaceX", "normal", "straight down"],
   help="How steeply they come in (flight-path angle below the horizontal): extreme 35-48 deg (very shallow, a long glide), intermediate 48-62, SpaceX 62-74 (about a real Falcon 9 landing-burn entry), "
        "straight down 87-90. normal = each level's own range (level 0: 72-85 deg ... level 3: 42-72 deg), the default.")
_f("traj_variation", "Start conditions", "Trajectory variation", "float", C.TRAJ_VARIATION, ("config", "TRAJ_VARIATION"), 0, 1, .05,
   help="How much the entry angle and entry speed change from episode to episode: 1 = the whole range, 0 = always the middle of it (every episode the same path).")
_f("spawn_height", "Start conditions", "Spawn height above the pad (m)", "float", C.SPAWN_HEIGHT_M, ("config", "SPAWN_HEIGHT_M"), 200, 6000, 50,
   help="3000 m is the real landing-burn start.")
_f("v_real_min", "Start conditions", "Entry speed at full level, min (m/s)", "float", C.SPACEX_V_MIN, ("config", "SPACEX_V_MIN"), 0, 1500, 10,
   help="The speed at spawn is drawn between min and max, times the level's speed fraction (35%, 60%, 80%, 100%).")
_f("v_real_max", "Start conditions", "Entry speed at full level, max (m/s)", "float", C.SPACEX_V_MAX, ("config", "SPACEX_V_MAX"), 0, 1500, 10)
_f("speed_scale", "Start conditions", "Entry speed scale", "float", C.SPACEX_SPEED_SCALE, ("config", "SPACEX_SPEED_SCALE"), 0, 3, .05,
   help="1.0 = the real entry speed. Lower makes it easier.")
_f("ramp_start", "Start conditions", "Extra speed ramp: speed at level 0 (fraction)", "float", 1.0, ("ramp", "start"), 0.05, 1, .05,
   help="The levels already ramp the entry speed (35%, 60%, 80%, 100%). This scales every level further: it starts at this fraction at level 0 and rises "
        "linearly to 100% at the last level. 1.0 = off.")
# ---- dual land (used when the start profile is dual)
_f("dual_spacing", "Dual land", "Pad spacing (m)", "float", C.DUAL_PAD_SPACING_M, ("config", "DUAL_PAD_SPACING_M"), 85, 6000, 5,
   help="Distance between the two pad centres. The landers start this far apart too (each on the same path, relative to its own pad).")
_f("dual_height", "Dual land", "Spawn height (m)", "float", C.DUAL_SPAWN_HEIGHT_M, ("config", "DUAL_SPAWN_HEIGHT_M"), 500, 8000, 100,
   help="Both landers start this high above their pads (4000 m by default).")
_f("dual_jitter", "Dual land", "Start differences (0 = identical paths)", "float", C.DUAL_JITTER, ("config", "DUAL_JITTER"), 0, 1, .05,
   help="Size of the small independent differences between the two starts (entry speed +-3% x this, tilt +-1 deg x this, spin, 8 m position). 0 = exactly the same path.")
_f("dual_tol", "Dual land", "Landing together: tolerance (s)", "float", C.DUAL_SYNC_TOL_S, ("config", "DUAL_SYNC_TOL_S"), 0.1, 20, .1,
   help="Touchdown time difference up to which the two count as landing together (joint success needs it).")
_f("dual_bonus", "Dual land", "Landing together: bonus (points)", "float", C.DUAL_SYNC_BONUS, ("config", "DUAL_SYNC_BONUS"), 0, 50_000, 50,
   help="Paid when both survive: bonus x (1 - time difference / tolerance).")
_f("dual_shape", "Dual land", "Height-gap shaping (points per m)", "float", C.DUAL_SYNC_SHAPE, ("config", "DUAL_SYNC_SHAPE"), 0, 10, .05,
   help="Reward for closing the height gap between the two landers while both are in the air (potential shaping: no drift).")
# ---- environment
_f("settle_s", "Environment", "Time to settle after touching down (s)", "float", C.SETTLE_S, ("config", "SETTLE_S"), 0, 5, .5,
   help="After the first foot touches the ground the lander gets this long to straighten up: the legs absorb the impact (spring + damper), the main engine shuts down at touchdown, "
        "and it is judged when it has been upright and nearly still for 0.6 s, or when the time is up. A touchdown that is already too fast, too sideways or tilted more than 0.45 rad is a crash at once. "
        "0 = the old rule: the episode ends the instant the first foot touches.")
_f("fuel_budget", "Environment", "Landing propellant loaded (kg)", "float", C.FUEL_BUDGET, ("config", "FUEL_BUDGET"), 500, 20_000, 50,
   help="Real propellant mass, burned at about 300 kg/s at full throttle. Calibrated by python -m src.sim.calibrate (1.5x the expert's median need).")
_f("side_fuel_cost", "Environment", "Side-engine fuel cost (x main)", "float", C.SIDE_FUEL_COST, ("config", "SIDE_FUEL_COST"), 0, 5, .05)
_f("tilt_limit", "Environment", "Tilt limit: terminate beyond (degrees from vertical)", "float", C.TILT_LIMIT_DEG, ("config", "TILT_LIMIT_DEG"), 5, 180, .5,
   help="Environment rule for BOTH reward types: at any time or altitude, a lander tilted more than this is terminated instantly. 180 = off.")
_f("tilt_spawn_max", "Environment", "Spawn tilt cap (degrees)", "float", C.TILT_SPAWN_MAX_DEG, ("config", "TILT_SPAWN_MAX_DEG"), 0, 90, 1,
   help="Start tilts are capped here so every start is legal under the limit (keep it clearly below the limit).")
_f("time_limit", "Environment", "Time limit floor to touch down (seconds)", "float", C.TIME_LIMIT_S, ("config", "TIME_LIMIT_S"), 0.5, 120, .5,
   help="Environment rule for BOTH reward types: both legs must touch within this long of the start, otherwise the episode is cut off (timeout penalty).")
_f("settle", "Environment", "Settle time after touchdown (seconds)", "float", C.SETTLE_S, ("config", "SETTLE_S"), 0, 30, .5,
   help="After touchdown the lander gets this long to come to rest (the simulator needs about 0.5 s of stillness). 0 = the time limit caps the whole episode.")
_f("max_ignitions", "Environment", "Maximum main-engine ignitions (0 = unlimited)", "int", C.MAX_IGNITIONS, ("config", "MAX_IGNITIONS"), 0, 1000,
   help="Environment rule for BOTH reward types: after this many ignitions the main engine cannot be lit again.")
_f("stall_speed", "Environment", "Stall: terminate when slower than (m/s)", "float", C.STALL_SPEED_MPS, ("config", "STALL_SPEED_MPS"), 0, 100, 1,
   help="Environment rule for BOTH reward types: airborne and slower than this = stalled = terminated at once. 0 = off.")
_f("stall_min_h", "Environment", "Stall: only above this height (m)", "float", C.STALL_MIN_HEIGHT_M, ("config", "STALL_MIN_HEIGHT_M"), 0, 2000, 10,
   help="The last stretch is exempt so a gentle touchdown is not a stall.")
_f("stall_grace", "Environment", "Stall: grace period after the start (steps, 20 = 1 s)", "int", C.STALL_GRACE_STEPS, ("config", "STALL_GRACE_STEPS"), 0, 500)
# ---- staged rules gate
_f("gate_rate", "Curriculum", "Staged rules: advance at survived-landing share", "float", C.RULES_GATE_RATE, ("config", "RULES_GATE_RATE"), .01, 1, .01,
   help="Share of the last N episodes at the current stage that end with the lander safely at rest.")
_f("gate_window", "Curriculum", "Staged rules: ...over the last N episodes", "int", C.RULES_GATE_WINDOW, ("config", "RULES_GATE_WINDOW"), 10, 5000)
_f("gate_max", "Curriculum", "Staged rules: advance anyway after N steps in a stage (0 = never)", "int", C.RULES_GATE_MAX_STEPS, ("config", "RULES_GATE_MAX_STEPS"), 0, 50_000_000)
# ---- curriculum
_f("advance_rate", "Curriculum", "Advance at success rate", "float", C.ADVANCE_SUCCESS_RATE, ("config", "ADVANCE_SUCCESS_RATE"), .05, 1, .05)
_f("advance_window", "Curriculum", "...over last N episodes", "int", C.ADVANCE_WINDOW, ("config", "ADVANCE_WINDOW"), 10, 5000)
_f("advance_max", "Curriculum", "Max steps per level", "int", C.ADVANCE_MAX_STEPS, ("config", "ADVANCE_MAX_STEPS"), 10_000, 20_000_000, 10_000)
# ---- PPO
_f("learning_rate", "PPO", "Learning rate", "float", C.PPO_KWARGS["learning_rate"], ("ppo", "learning_rate"), 1e-6, 1e-2, 1e-4)
_f("n_steps", "PPO", "n_steps (per env)", "int", C.PPO_KWARGS["n_steps"], ("ppo", "n_steps"), 64, 16384)
_f("batch_size", "PPO", "Batch size", "int", C.PPO_KWARGS["batch_size"], ("ppo", "batch_size"), 8, 65536,
   help="Must divide n_steps x environments")
_f("gamma", "PPO", "Gamma", "float", C.PPO_KWARGS["gamma"], ("ppo", "gamma"), .9, 1, .001)
_f("gae_lambda", "PPO", "GAE lambda", "float", C.PPO_KWARGS["gae_lambda"], ("ppo", "gae_lambda"), .8, 1, .01)
_f("ent_coef", "PPO", "Entropy coefficient", "float", C.PPO_KWARGS["ent_coef"], ("ppo", "ent_coef"), 0, .2, .005)
# ---- reward (custom only)
_f("r_scale", "Custom reward", "Global scale", "float", R.GLOBAL_SCALE, ("reward", "GLOBAL_SCALE"), 1e-4, 10, .001,
   help="Applied to the total reward. Logged term values stay unscaled.")
_f("r_w_dist", "Custom reward", "Shaping: distance to pad (per m)", "float", R.W_DIST, ("reward", "W_DIST"), 0, 1000, 1,
   help="Potential-based: paid for the CHANGE in distance, so it cannot be farmed by looping.")
_f("r_w_vel", "Custom reward", "Shaping: speed above safe envelope (per m/s)", "float", R.W_VEL, ("reward", "W_VEL"), 0, 1000, 1,
   help="The braking signal. Falling faster than the safe speed for the current height costs this much per m/s.")
_f("r_env_decel", "Custom reward", "Envelope: planned braking (m/s^2)", "float", R.ENVELOPE_DECEL, ("reward", "ENVELOPE_DECEL"), 1, 30, .5,
   help="Lower = brake earlier. The lander can deliver about 8.9 m/s^2 at full throttle (measured).")
_f("r_env_floor", "Custom reward", "Envelope: touchdown speed allowed (m/s)", "float", R.ENVELOPE_FLOOR, ("reward", "ENVELOPE_FLOOR"), 0, 10, .1,
   help="Measured: touchdowns at or below 1.5 m/s survive 98% of the time.")
_f("r_w_att", "Custom reward", "Shaping: tilt (per rad)", "float", R.W_ATT, ("reward", "W_ATT"), 0, 1000, 5)
_f("r_w_spin", "Custom reward", "Shaping: spin (per rad/s)", "float", R.W_SPIN, ("reward", "W_SPIN"), 0, 1000, 1)
_f("r_w_leg", "Custom reward", "Shaping: per leg on the ground", "float", R.W_LEG, ("reward", "W_LEG"), 0, 1000, 5)
_f("r_time", "Custom reward", "Time penalty (per step)", "float", R.TIME_PENALTY, ("reward", "TIME_PENALTY"), -10, 0, .01)
_f("r_fuel", "Custom reward", "Fuel cost (per unit)", "float", R.FUEL_PER_UNIT, ("reward", "FUEL_PER_UNIT"), -10, 0, .01)
_f("r_slow_rate", "Custom reward", "Slow approach: points/s for a stopped lander at 10 m", "float", R.SLOW_RATE_REF, ("reward", "SLOW_RATE_REF"), 0, 1000, .5,
   help="Charged while the lander is slower than the speed floor for its height. The rate grows exponentially with altitude (see 'growth').")
_f("r_slow_growth", "Custom reward", "Slow approach: rate multiplier per 30 m of altitude", "float", R.SLOW_GROWTH, ("reward", "SLOW_GROWTH"), 1, 10_000, 5,
   help="100 = about 2/s at 10 m, 200/s at 40 m for a stopped lander. 1 = same charge at every height.")
_f("r_slow_limit", "Custom reward", "Slow approach: speed floor (fraction of the safe envelope)", "float", R.SLOW_LIMIT_FRAC, ("reward", "SLOW_LIMIT_FRAC"), 0, 1, .05,
   help="Floor = this x the safe-descent speed at the current height (about 18 m/s at 40 m, 9 m/s at 10 m for 0.75). Higher = demands a later, harder burn.")
_f("r_slow_min_h", "Custom reward", "Slow approach: exempt below (m)", "float", R.SLOW_MIN_HEIGHT, ("reward", "SLOW_MIN_HEIGHT"), 0, 2000, .5)
_f("r_ascend", "Custom reward", "Ascent penalty (once per episode)", "float", R.ASCEND_PENALTY, ("reward", "ASCEND_PENALTY"), -100_000, 0, 50)
_f("r_ascend_speed", "Custom reward", "Ascent: upward speed that counts (m/s)", "float", R.ASCEND_SPEED, ("reward", "ASCEND_SPEED"), 0, 20, .5)
_f("r_offcourse", "Custom reward", "Off-course charge (points per second)", "float", R.OFFCOURSE_PER_SEC, ("reward", "OFFCOURSE_PER_SEC"), -1000, 0, 1,
   help="Charged while the predicted touchdown point (if the engines cut out now) is outside the tolerance. It teaches lateral control: braking lengthens the fall, so sideways speed must be cancelled too.")
_f("r_offcourse_tol", "Custom reward", "Off-course: tolerance around the pad centre (m)", "float", R.OFFCOURSE_TOLERANCE_M, ("reward", "OFFCOURSE_TOLERANCE_M"), 0, 500, .5)
_f("r_tilt_penalty", "Custom reward", "Tilt-out penalty", "float", R.TILT_OUT_PENALTY, ("reward", "TILT_OUT_PENALTY"), -100_000, 0, 50,
   help="Paid once when the lander is terminated for tilting past the limit.")
_f("r_tilt_projected", "Custom reward", "Tilt-out: also score the crash it was heading for (1 = yes, 0 = flat)", "float", R.TILT_OUT_PROJECTED, ("reward", "TILT_OUT_PROJECTED"), 0, 1, 1,
   help="Flat is exploitable: dying early (-1000) beats a free fall (about -1850), so the lander could learn to flip itself on purpose.")
_f("r_first_fire", "Custom reward", "First ignition of the main engine: bonus", "float", R.FIRST_IGNITION_BONUS, ("reward", "FIRST_IGNITION_BONUS"), 0, 100_000, 10,
   help="Paid once per episode, when the main engine is lit for the first time. Re-ignitions then cost the amounts below.")
_f("r_refire_base", "Custom reward", "Re-ignition: cost of the 2nd ignition (the 1st is free)", "float", R.REFIRE_BASE, ("reward", "REFIRE_BASE"), 0, 100_000, 5,
   help="Every further ignition multiplies the cost by the growth factor below, up to the ignition limit (default 10: 25, 50, 100 ... 6400).")
_f("r_refire_growth", "Custom reward", "Re-ignition: cost multiplier per extra ignition", "float", R.REFIRE_GROWTH, ("reward", "REFIRE_GROWTH"), 1, 10, .1)
_f("r_stall_penalty", "Custom reward", "Stall penalty (zero tolerance)", "float", R.STALL_OUT_PENALTY, ("reward", "STALL_OUT_PENALTY"), -100_000, 0, 50,
   help="Paid once when a lander is terminated for stalling in the air.")
_f("r_stall_projected", "Custom reward", "Stall: also score the crash it was heading for (1 = yes, 0 = flat)", "float", R.STALL_OUT_PROJECTED, ("reward", "STALL_OUT_PROJECTED"), 0, 1, 1,
   help="Flat is exploitable: ending the episode by stalling (-1500) would beat a free-fall crash (about -1850).")
_f("r_land_off", "Custom reward", "Landing: survived outside the green zone", "float", R.LAND_OFF_PAD, ("reward", "LAND_OFF_PAD"), 0, 100_000, 10)
_f("r_land_zone", "Custom reward", "Landing: survived inside the green zone", "float", R.LAND_ZONE, ("reward", "LAND_ZONE"), 0, 100_000, 10)
_f("r_land_edge", "Custom reward", "Landing: soft landing at the limits (zone edge)", "float", R.LAND_SOFT_EDGE, ("reward", "LAND_SOFT_EDGE"), 1, 100_000, 10,
   help="Soft = in the zone, <= 1.5 m/s, <= 0.15 rad. The score rises exponentially with accuracy from this value.")
_f("r_land_perfect", "Custom reward", "Landing: perfect bullseye", "float", R.LAND_PERFECT, ("reward", "LAND_PERFECT"), 1, 1_000_000, 100,
   help="Bullseye, 0 m/s and perfectly upright. Score = edge x (perfect/edge)^accuracy.")
_f("r_soft_speed", "Custom reward", "Soft landing: max touchdown speed (m/s)", "float", R.SOFT_SPEED, ("reward", "SOFT_SPEED"), .1, 5, .1)
_f("r_soft_tilt", "Custom reward", "Soft landing: max tilt (rad)", "float", R.SOFT_TILT, ("reward", "SOFT_TILT"), .01, 1, .01)
_f("r_crash_in_edge", "Custom reward", "Crash position: inside the zone, at the edge", "float", R.CRASH_IN_EDGE, ("reward", "CRASH_IN_EDGE"), -100_000, 100_000, 10,
   help="Position score of a crash. Inside the green zone it rises exponentially from this value at the edge to the centre value.")
_f("r_crash_in_center", "Custom reward", "Crash position: inside the zone, dead centre", "float", R.CRASH_IN_CENTER, ("reward", "CRASH_IN_CENTER"), -100_000, 100_000, 10)
_f("r_crash_out_edge", "Custom reward", "Crash position: just outside the zone", "float", R.CRASH_OUT_EDGE, ("reward", "CRASH_OUT_EDGE"), -100_000, 0, 10,
   help="Starts here at the zone edge and grows exponentially with every metre of miss.")
_f("r_crash_out_growth", "Custom reward", "Crash position: outside penalty multiplier per metre", "float", R.CRASH_OUT_GROWTH, ("reward", "CRASH_OUT_GROWTH"), 1, 5, .05)
_f("r_crash_out_cap", "Custom reward", "Crash position: outside penalty floor", "float", R.CRASH_OUT_CAP, ("reward", "CRASH_OUT_CAP"), -1_000_000, 0, 100)
_f("r_crash_speed_goal", "Custom reward", "Crash speed: goal (m/s)", "float", R.CRASH_SPEED_GOAL, ("reward", "CRASH_SPEED_GOAL"), 0, 10, .1,
   help="Speed score of a crash is independent of the position score and is added to it.")
_f("r_crash_speed_per", "Custom reward", "Crash speed: points per m/s above the goal", "float", R.CRASH_SPEED_PER_MPS, ("reward", "CRASH_SPEED_PER_MPS"), -10_000, 0, 10)
_f("r_crash_fade", "Custom reward", "Crash position bonus: fades to 0 over this overspeed (m/s, 0 = no fade)", "float", R.CRASH_POS_FADE_MPS, ("reward", "CRASH_POS_FADE_MPS"), 0, 100, 1,
   help="A centred crash only earns the position bonus if it is also slow: the bonus is multiplied by 1 - overspeed / this. Penalties outside the zone are not reduced.")
_f("r_crash_pen", "Custom reward", "Crash: flat independent penalty", "float", R.CRASH_PENALTY, ("reward", "CRASH_PENALTY"), -100_000, 0, 10,
   help="Added to every crash or out-of-bounds on top of the position and speed scores: a light push not to crash.")
_f("r_crash_vert", "Custom reward", "Crash in zone, near vertical: extra bonus (0 = off)", "float", R.CRASH_VERTICAL_BONUS, ("reward", "CRASH_VERTICAL_BONUS"), 0, 100_000, 10)
_f("r_vertical_deg", "Custom reward", "Near vertical means within (degrees)", "float", R.VERTICAL_DEG, ("reward", "VERTICAL_DEG"), 0, 90, .5)
_f("r_timeout", "Custom reward", "Timeout penalty (never landed)", "float", R.TIMEOUT_PENALTY, ("reward", "TIMEOUT_PENALTY"), -100_000, 0, 10)
_f("r_fuel_out", "Custom reward", "Out-of-fuel penalty", "float", R.FUEL_OUT_PENALTY, ("reward", "FUEL_OUT_PENALTY"), -100_000, 0, 10)

BY_KEY = {f["key"]: f for f in F}


def public_schema() -> list[dict]:
    return [{k: v for k, v in f.items() if k not in ("target", "always")} for f in F]


def build_run(params: dict) -> dict:
    """Validate form values -> dict(name, argv, overrides) or dict(errors=[...])."""
    errors, vals = [], {}
    for f in F:
        raw = params.get(f["key"], f["default"])
        try:
            if f["type"] == "int":
                v = int(float(raw))
            elif f["type"] == "float":
                v = float(raw)
            elif f["type"] == "select":
                v = str(raw)
                if v not in f["options"]:
                    raise ValueError(f"must be one of {f['options']}")
            else:
                v = str(raw)
                if not re.fullmatch(r"[A-Za-z0-9_\-]{0,60}" if f["key"] == "init_from" else r"[A-Za-z0-9_\-]{0,24}", v):
                    raise ValueError("letters, digits, _ and - only (max 24)" if f["key"] != "init_from" else "letters, digits, _ and - only (max 60)")
            if f["type"] in ("int", "float"):
                if f["min"] is not None and v < f["min"]:
                    raise ValueError(f"min {f['min']}")
                if f["max"] is not None and v > f["max"]:
                    raise ValueError(f"max {f['max']}")
        except (ValueError, TypeError) as e:
            errors.append(f"{f['label']}: {e}")
            continue
        vals[f["key"]] = v
    if errors:
        return dict(errors=errors)
    sweep = vals["sweep"] == "on"
    if vals["v_real_min"] > vals["v_real_max"]:
        return dict(errors=["SpaceX real speed min must be <= max"])
    if sweep and vals["n_envs"] % vals["sweep_copies"]:
        return dict(errors=[f"Parallel environments ({vals['n_envs']}) must be a multiple of copies per learning rate ({vals['sweep_copies']})"])
    rollout = vals["n_steps"] * (1 if sweep else vals["n_envs"])      # sweep agents have one env each
    if rollout % vals["batch_size"]:
        return dict(errors=[f"Batch size {vals['batch_size']} must divide n_steps x environments = {rollout}"
                            + (" (each sweep agent has one environment)" if sweep else "")])
    groups = vals["n_envs"] // vals["sweep_copies"] if sweep else 0
    if sweep and max(C.agent_lrs(vals["learning_rate"], vals["lr_growth"], groups)) > C.MAX_LEARNING_RATE:
        return dict(errors=[f"Sweep learning rates reach {max(C.agent_lrs(vals['learning_rate'], vals['lr_growth'], groups)):.3g}; "
                            f"lower the base rate, growth or number of environments (max {C.MAX_LEARNING_RATE})"])
    if vals["init_from"] and sweep:
        return dict(errors=["Continue-from-checkpoint works on a shared-policy run: turn the learning-rate sweep off"])
    if vals["init_from"] and not (C.CHECKPOINT_DIR / f"{vals['init_from']}.zip").exists() and not all((C.CHECKPOINT_DIR / f"{vals['init_from']}_{s}.zip").exists() for s in "AB"):
        return dict(errors=[f"No checkpoint named {vals['init_from']} (see the Data tab for the list)"])
    if vals["stop_at_level"] >= 0 and vals["stop_at_level"] < vals["start_level"]:
        return dict(errors=["Stop level must be >= start level"])

    if vals["tag"] == "_app" and vals["start_profile"] != "spacex":       # the default tag follows the profile, so a dual / pair run does not overwrite the single-lander run
        vals["tag"] = "_" + vals["start_profile"]
    argv, overrides = [sys.executable, "src/training/train.py"], dict(config={}, reward={})
    ppo = {}
    for f in F:
        v, (kind, name) = vals[f["key"]], f["target"]
        if kind == "arg":
            if (f["key"] == "stop_at_level" and v < 0) or (f["key"] == "init_from" and not v):
                continue
            argv += [name, str(v)]
        elif kind in ("sweep", "ramp", "staged"):
            continue
        elif kind == "config" and (v != f["default"] or f["always"]):
            overrides["config"][name] = v
        elif kind == "ppo" and v != f["default"]:
            ppo[name] = v
        elif kind == "reward" and v != f["default"] and vals["reward"] == "custom":
            overrides["reward"][name] = v
    if ppo:
        overrides["config"]["PPO_KWARGS"] = ppo
    overrides["config"]["RULES_STAGE_START"] = 0 if vals["staged"] == "on" else C.RULES_STAGE_MAX
    r0 = vals["ramp_start"]
    if r0 < 1.0:
        n = C.N_LEVELS
        overrides["config"]["SPACEX_LEVEL_SCALE"] = [round(r0 + (1.0 - r0) * i / (n - 1), 4) for i in range(n)]
    name = f"{vals['reward']}_s{vals['seed']}{vals['tag']}"
    if sweep:
        argv[1] = "src/training/sweep.py"
        argv += ["--lr-growth", str(vals["lr_growth"]), "--copies", str(vals["sweep_copies"])]
        name += "_sweep"
    return dict(name=name, argv=argv, overrides=overrides, values=vals)
