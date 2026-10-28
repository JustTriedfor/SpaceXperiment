"""All tunable numbers for SpaceXperiment, in one place.

Real-scale Falcon 9 landing burn: metres, kilograms, seconds, newtons, radians. The physics is our own 2-D engine
(src/sim), not Box2D. Vehicle numbers live in src/sim/vehicle.py; this file holds the task, the rules and the training setup.
"""
from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT / "assets"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
MODEL_CONFIG_DIR = ROOT / "models" / "configs"
LOG_DIR = ROOT / "logs"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# --------------------------------------------------------------------------
# Units and the world
# --------------------------------------------------------------------------
FPS = 20                                       # agent decisions per second (the physics runs PHYSICS_SUBSTEPS times faster inside each)
PHYSICS_SUBSTEPS = 5                           # 0.01 s physics step
GRAVITY_MPS2 = 9.81
PAD_HALF_WIDTH_M = 45.0                        # a droneship deck is about 90 m wide: the pad is drawn 90 m wide, centred at x = 0
LEG_DOWN_M = 15.0                              # centre of gravity -> foot level when upright (CG 14 m above the base, feet 1 m below it)
START_X_M = 0.0                                # the pad centre is x = 0 in this version
START_Y_M = 3000.0                             # nominal spawn altitude above the pad

# observation scales (obs = value / scale); the policy sees [x, h, vx, vy, tilt, spin, ...] (see src/sim/env.py)
OBS_X_TO_M = 2000.0
OBS_Y_TO_M = 3000.0
OBS_VX_TO_MPS = 300.0
OBS_VY_TO_MPS = 300.0
OBS_SPIN_TO_RAD_S = 0.5
OBS_Q_TO_PA = 50_000.0


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
MAX_EPISODE_STEPS = 2400                       # hard cap (120 s)

# Curriculum. speed = fraction of the full re-entry speed, entry_deg = flight-path angle below the horizontal (min, max), miss_m = how far the
# ballistic impact point may be from the pad, aoa_deg = random angle of attack at the start, wind in m/s, ang_vel in rad/s.
LEVELS = [
    dict(wind_power=0.0,  turbulence_power=0.0, tilt=math.radians(10), speed=0.35, x_offset=50.0,  ang_vel=0.02, entry_deg=(72.0, 85.0), aoa_deg=3.0),
    dict(wind_power=5.0,  turbulence_power=0.5, tilt=math.radians(15), speed=0.60, x_offset=120.0, ang_vel=0.04, entry_deg=(62.0, 82.0), aoa_deg=6.0),
    dict(wind_power=10.0, turbulence_power=1.0, tilt=math.radians(20), speed=0.80, x_offset=220.0, ang_vel=0.06, entry_deg=(52.0, 78.0), aoa_deg=10.0),
    dict(wind_power=15.0, turbulence_power=1.5, tilt=math.radians(25), speed=1.00, x_offset=350.0, ang_vel=0.08, entry_deg=(42.0, 72.0), aoa_deg=15.0),
]
N_LEVELS = len(LEVELS)

# Spawn. 'spacex' = the re-entry side entry: SPAWN_HEIGHT_M above the pad at SPACEX_V_MAX m/s (level speed fraction x that), on a path found by
# simulating the passive descent (drag included) so that it would hit near the pad. 'classic' is kept only as a name for the old table.
START_PROFILE = "spacex"
SPAWN_HEIGHT_M = 3000.0
# Re-entry angle (flight-path angle below the horizontal at the start) and how much it varies from episode to episode.
# ENTRY_PRESET: "normal" = the level table above (each level has its own range; the default), or one of the fixed ranges below, whatever the level.
# TRAJ_VARIATION: 1 = the whole range (and the whole entry-speed range) is used, 0 = always the middle of it, in between = a narrower band around the middle.
ENTRY_PRESETS = {                                  # degrees below the horizontal (min, max); the order from shallowest to steepest is the order of the names
    "extreme": (35.0, 48.0),                       # very shallow: a long sideways glide, lots of distance to cover
    "intermediate": (48.0, 62.0),
    "spacex": (62.0, 74.0),                        # about what a Falcon 9 booster's landing-burn entry looks like
    "straight": (87.0, 90.0),                      # almost straight down
}
ENTRY_PRESET = "normal"
TRAJ_VARIATION = 1.0
SPACEX_V_MIN = 250.0                           # entry speed range at full level (m/s); the speed is drawn uniformly in [V_MIN, V_MAX] x level fraction
SPACEX_V_MAX = 320.0
SPACEX_SPEED_SCALE = 1.0
SPACEX_LATERAL_M = [50.0, 120.0, 220.0, 350.0]   # kept for the app: the ballistic-impact miss allowance per level (see LEVELS x_offset)
SPACEX_AOA_DEG = [3.0, 6.0, 10.0, 15.0]
SPACEX_LEVEL_SCALE = [1.0, 1.0, 1.0, 1.0]      # extra factor on each level's speed fraction; the app ramps it from `ramp_start` to 1.0

# Start profile 'dual' (Dual land): ONE environment with TWO landers and TWO pads. Both start together DUAL_SPAWN_HEIGHT_M up on the same
# predetermined path (the pad-relative start is identical, shifted by the pad spacing, with a small independent jitter), steered by one policy
# (actions and observations of both landers side by side), and the episode ends when both are down. The goal is to land at about the same time.
DUAL_PAD_SPACING_M = 1000.0                    # distance between the two pad centres (adjustable); the landers start this far apart too
DUAL_SPAWN_HEIGHT_M = 4000.0
DUAL_JITTER = 0.3                              # 0..1: size of the small independent differences between the two starts (0 = exactly the same path)
DUAL_SYNC_TOL_S = 2.0                          # landing-time difference up to which they count as landing together
DUAL_SYNC_BONUS = 1000.0                       # points when both survive: x (1 - |dt| / tolerance), 0 beyond it
DUAL_SYNC_SHAPE = 0.5                          # points per metre of closing the height gap between the two landers (potential shaping, drift free)
DUAL_OBS_DIM = 2 * 18 + 4                      # two lander observations + [height gap, fall-speed gap, A down, B down]
DUAL_ACT_DIM = 2 * 5

# Fuel is the real propellant mass: FUEL_BUDGET is the landing propellant loaded (kg). Calibrated by src/sim/calibrate.py.
SIDE_FUEL_COST = 0.0
FUEL_BUDGET = 4500.0

# Success definition (evaluation + curriculum gating)
SUCCESS_MAX_DIST_M = 10.0
SUCCESS_MAX_SPEED_MPS = 3.0                    # the legs absorb a few m/s (the best real landings touch down at 1-2 m/s; a landing at up to 3 m/s counts as soft here)
SUCCESS_MAX_TILT_RAD = 0.15
SURVIVE_MAX_VSPEED_MPS = 5.0                   # touching down faster than this (or too tilted / too sideways) destroys the booster
SURVIVE_MAX_HSPEED_MPS = 3.0
SURVIVE_MAX_TILT_RAD = 0.20

# After the first foot touches the ground the lander gets SETTLE_S seconds to settle: the legs absorb the impact (spring + damper, see vehicle.py),
# the main engine shuts down at touchdown (as on a real booster), and RCS can still help it straighten up. It is judged when it has been calm for
# SETTLE_HOLD_S (upright, nearly still) or when the time is up: upright enough (SURVIVE_MAX_TILT_RAD) and nearly still = rest, else a crash.
# A touchdown that is already too fast / sideways / tilted (CONTACT_TILT_MAX) is a crash at once, and so is tipping over (TIP_OVER_RAD).
# SETTLE_S = 0 restores the old rule: the episode ends the instant the first foot touches.
SETTLE_S = 2.5
SETTLE_HOLD_S = 0.6
SETTLE_REST_SPEED = 0.5        # m/s: "nearly still" (the final check allows three times this)
SETTLE_REST_SPIN = 0.05        # rad/s
CONTACT_TILT_MAX = 0.45        # rad of tilt at first contact beyond which it is a crash straight away
TIP_OVER_RAD = 0.62            # rad: past this the stage falls over (legs 18 m apart, CG 15 m up: it tips at about 0.54)
TOUCHDOWN_ENGINE_CUT = True
# Environment rule: tilted more than this from vertical, at ANY time, is terminated. (The re-entry starts tilted to the flight path.)
TILT_LIMIT_DEG = 75.0
TILT_SPAWN_MAX_DEG = 60.0
# Environment rules, staged (see below). The clock is at least TIME_LIMIT_SCALE x the fastest possible soft landing from the spawn
# (coast, then brake at the engine's limit) plus TIME_LIMIT_PAD_S; TIME_LIMIT_S is the floor.
TIME_LIMIT_SCALE = 1.5
TIME_LIMIT_PAD_S = 8.0
TIME_LIMIT_S = 30.0
MAX_IGNITIONS = 3                              # a Merlin 1D can be restarted a few times; the landing burn is one
ALT_BANDS_M = (30.0, 80.0, 160.0, 320.0, 640.0, 1200.0, 2200.0)   # altitude band edges for the learning heat-map (8 bands)
BRAKE_MAX_MPS2 = 23.0                          # net braking at full throttle (T/W about 3.3 at a 25 t landing mass, minus g); refined by calibrate.py
STALL_SPEED_MPS = 6.0                          # airborne and slower than this (above STALL_MIN_HEIGHT_M) = stalled = terminated
STALL_MIN_HEIGHT_M = 100.0
STALL_GRACE_STEPS = 60                         # 3 s after the start are exempt
# Staged rules as before: stage 0 = tilt limit + shaping + scores, stage 1 = + the clock and the stall rule, stage 2 = + the ignition limit.
RULES_STAGE_START = 2
RULES_STAGE_MAX = 2
RULES_GATE_RATE = 0.20
RULES_GATE_WINDOW = 200
RULES_GATE_MAX_STEPS = 1_000_000
RULES_GATE_CAP_MIN_RATE = 0.10  # the step cap only advances a stage if at least this share of episodes survived; below it the stage HOLDS
GATE_COOLDOWN_STEPS = 150_000   # after any level / stage advance, neither gate may fire again for this many steps (no two harder things at once)
GATE_MAX_SPEED_MPS = 3.0       # curriculum gate: a 'soft landing' uses the same speed limit as the strict success

# --------------------------------------------------------------------------
# Curriculum schedule
# --------------------------------------------------------------------------
ADVANCE_SUCCESS_RATE = 0.80
ADVANCE_WINDOW = 200
ADVANCE_MAX_STEPS = 500_000
ADVANCE_CAP_MIN_RATE = 0.10     # the step cap only advances a level if the soft-landing rate over the window is at least this; below it the level HOLDS

# --------------------------------------------------------------------------
# PPO
# --------------------------------------------------------------------------
N_ENVS = 8
# Record the policy network's activations with every streamed live episode (feeds the app's Brain tab).
LIVE_ACTIVATIONS = True
# Step-level trajectory logging for the app's Data tab (experience that can be exported and cloned from).
# Kept: every DATA_EVERY-th episode plus the DATA_TAIL best and DATA_TAIL worst of the recent episodes (by return). Compressed CSV chunks
# under data/trajectories/<run>/. When all trajectory data together exceed DATA_BUDGET_GB the oldest chunks are deleted.
DATA_LOG = True
DATA_EVERY = 10
DATA_TAIL = 0.05
DATA_BUDGET_GB = 3.0
# Where the neural network trains. The simulation always runs on the CPU. 'cpu' | 'cuda' | 'auto' (cuda if this PyTorch can use it).
DEVICE = "cpu"
TOTAL_TIMESTEPS = 5_000_000
SEEDS = (0, 1, 2)
MAX_LEARNING_RATE = 0.1


def agent_lrs(base: float, growth: float, n: int) -> list[float]:
    """Sweep mode: learning rate of agent i is base * growth**i (exponentially increasing)."""
    return [base * growth ** i for i in range(n)]


NET_ARCH = [256, 256]          # hidden layers of the actor and of the critic (the old project used 64 x 64)
NET_ARCH_DUAL = [512, 512]     # the dual-lander policy is twice as wide (it holds two copies of the single-lander one side by side)


def net_arch() -> list:
    return list(NET_ARCH_DUAL if START_PROFILE == "dual" else NET_ARCH)      # 'pair' (two networks) uses the normal size for each
PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=512,            # measured: about 2.3x the throughput of 64 on this machine (see RUNBOOK)
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.005,
)


# --------------------------------------------------------------------------
# Optional overrides from the web app / command line. They travel in an environment variable so
# that spawned SubprocVecEnv workers (which re-import this module) see the same values.
#   ROCKETEER_OVERRIDES='{"config": {"FUEL_BUDGET": 4000, "PPO_KWARGS": {"ent_coef": 0.02}}, "reward": {...}}'
# --------------------------------------------------------------------------
def _apply_overrides() -> dict:
    import json
    import os

    raw = os.environ.get("ROCKETEER_OVERRIDES")
    if not raw:
        return {}
    data = json.loads(raw)
    for key, value in data.get("config", {}).items():
        if key == "PPO_KWARGS":
            PPO_KWARGS.update(value)
        elif key in globals():
            globals()[key] = value
    return data


OVERRIDES = _apply_overrides()
