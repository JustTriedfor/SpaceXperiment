"""The SpaceXperiment environment: spawn, observation, actions, touchdown outcomes, determinism, and the calibration numbers."""
import math
import time

import numpy as np
import pytest

from src.sim import vehicle as V
from src.sim.env import ACT_DIM, OBS_DIM, SpaceXEnv, throttle_from_action
from src.sim.expert import rollout
from src.sim.spawn import Wind, make_spawn
from src.utils import config as C

IDLE = np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def test_spaces_and_the_first_observation():
    env = SpaceXEnv(level=0)
    obs, info = env.reset(seed=0)
    assert env.observation_space.shape == (OBS_DIM,) == (18,) and env.action_space.shape == (ACT_DIM,) == (5,)
    assert obs.shape == (18,) and obs.dtype == np.float32
    assert obs[1] == pytest.approx(C.SPAWN_HEIGHT_M / C.OBS_Y_TO_M, rel=0.01)           # 3 km up
    assert obs[3] < 0 and obs[6] == 1.0 and obs[7] == 0.0 and obs[11] == 0.0 and obs[12] == 0.0     # falling, full tank, engine off, no ignitions
    assert np.all(np.abs(obs) < 20)
    assert info["time_limit_s"] >= C.TIME_LIMIT_S


def test_the_throttle_action_maps_to_off_or_the_40_to_100_percent_range():
    assert throttle_from_action(-1.0) == 0.0 and throttle_from_action(-0.5) == 0.0
    assert throttle_from_action(-0.49) == pytest.approx(V.THROTTLE_MIN, abs=0.01)
    assert throttle_from_action(1.0) == 1.0 and throttle_from_action(0.25) == pytest.approx(V.THROTTLE_MIN + 0.6 * 0.5)


def test_the_spawn_is_a_drag_simulated_side_entry_aimed_near_the_pad():
    rng = np.random.default_rng(0)
    for level in range(C.N_LEVELS):
        cfg = C.LEVELS[level]
        for _ in range(5):
            s, d = make_spawn(rng, level, prop=4500.0, wind=Wind(cfg["wind_power"], rng))
            assert s.foot_height() == pytest.approx(C.SPAWN_HEIGHT_M, abs=0.1)
            speed = math.hypot(s.vx, s.vy)
            assert C.SPACEX_V_MIN * cfg["speed"] * 0.999 <= speed <= C.SPACEX_V_MAX * cfg["speed"] * 1.001
            assert s.vy < 0 and abs(s.vx) > 1.0 and abs(s.x) > 100.0                     # it comes in from the side, descending
            assert abs(d["target_miss_m"]) <= cfg["x_offset"] + 1e-6
            assert abs(s.th) <= math.radians(C.TILT_SPAWN_MAX_DEG) + 1e-9                   # engine-first: the nose points against the airflow


def test_full_speed_level_is_about_300_m_per_s():
    rng = np.random.default_rng(1)
    speeds = [make_spawn(rng, C.N_LEVELS - 1, prop=4500.0)[1]["speed"] for _ in range(60)]
    assert 250 <= min(speeds) and max(speeds) <= 320 and 280 < float(np.mean(speeds)) < 290


def test_same_seed_same_episode_and_different_seeds_differ():
    def run(seed):
        env = SpaceXEnv(level=1)
        env.reset(seed=seed)
        out = []
        for _ in range(60):
            o, r, *_ = env.step(np.array([0.2, 0.1, -0.1, 0.0, 0.0], np.float32))
            out.append((float(o[0]), float(o[1]), r))
        return out
    assert run(4) == run(4) and run(4) != run(5)


def force_contact(env, vy, vx=0.0, th=0.0):
    """Put the booster just above the ground so that the next step touches down."""
    s = env.s
    s.th, s.om, s.vx, s.vy, s.x = th, 0.0, vx, vy, 3.0
    s.y = s.y - s.foot_height() + 0.04                                              # the lower foot is 4 cm above the ground
    s.on, s.thr = False, 0.0


def touchdown(vy, vx=0.0, th=0.0, x=3.0, om=0.0, record=False):
    """Drop the stage onto the pad and fly (engine off) until the episode ends. -> (terminated, last info, steps)"""
    env = SpaceXEnv(level=0, record=record)
    env.set_rules_stage(2)
    env.reset(seed=2)
    force_contact(env, vy, vx, th)
    env.s.x, env.s.om = x, om
    n = 0
    while True:
        obs, r, term, trunc, info = env.step(IDLE)
        n += 1
        if term or trunc or n > 200:
            return term, info, n


def test_touchdown_ends_the_episode_as_rest_or_crash():
    term, info, n = touchdown(-2.0)
    m = info["episode_metrics"]
    assert term and m["end"] == "rest" and m["touchdown"] and m["success"] and m["soft"] and m["merit"] > 50
    assert info["touchdown"]["speed_mps"] == pytest.approx(2.0, abs=0.5) and 5 < n <= int(C.SETTLE_S * C.FPS) + 2                  # it settled on its legs and was judged then
    for kwargs in (dict(vy=-30.0), dict(vy=-2.0, vx=6.0), dict(vy=-2.0, th=0.6)):      # too fast, too sideways, too tilted at the first touch: a crash at once
        term, info, n = touchdown(**kwargs)
        assert term and n == 1 and info["episode_metrics"]["end"] == "crash" and not info["episode_metrics"]["success"] and info["episode_metrics"]["merit"] == 0
    term, info, n = touchdown(-2.0, x=C.SUCCESS_MAX_DIST_M * 4)                              # survived but off the pad: not a success
    assert info["episode_metrics"]["end"] == "rest" and not info["episode_metrics"]["success"]


def test_a_tilted_touchdown_gets_time_to_straighten_up_on_its_legs():
    term, info, n = touchdown(-2.0, th=0.2, record=True)
    m = info["episode_metrics"]
    assert term and m["end"] == "rest" and m["success"] and m["tilt_rad"] < 0.15                         # touched down at 0.2 rad, ended upright
    frames = info["trajectory"]["frames"]
    assert n > 20 and max(f["a"] for f in frames) == pytest.approx(0.2, abs=0.02) and abs(frames[-1]["a"]) < 0.15       # the episode kept running and the angle came down
    assert max(max(f["lc"]) for f in frames) > 0.1 and max(max(f["lc"]) for f in frames) <= 0.9 + 1e-9                     # the legs gave (compressed) but did not bottom out
    assert all(f["on"] == 0 and f["m"] == 0.0 for f in frames[-10:])                                         # the engine shut down at touchdown


def test_a_stage_that_tips_over_after_touching_down_is_a_crash():
    term, info, n = touchdown(-2.0, th=0.4, om=0.5)                                                     # touched down upright enough, but rotating fast
    assert term and info["episode_metrics"]["end"] == "crash" and not info["episode_metrics"]["success"]


def test_settle_time_zero_restores_the_old_instant_verdict(monkeypatch):
    monkeypatch.setattr(C, "SETTLE_S", 0.0)
    term, info, n = touchdown(-2.0)
    assert term and n == 1 and info["episode_metrics"]["end"] == "rest"


def test_tilt_limit_and_out_of_bounds_end_the_episode():
    env = SpaceXEnv(level=0, fuel_budget=1e9)
    env.reset(seed=3)
    env.s.th = math.radians(C.TILT_LIMIT_DEG) + 0.05
    _, _, term, _, info = env.step(IDLE)
    assert term and info["tilt_out"] and info["episode_metrics"]["end"] == "tilt"
    env.reset(seed=3)
    env.s.x = 9500.0
    _, _, term, _, info = env.step(IDLE)
    assert term and info["episode_metrics"]["end"] == "oob"


def test_metrics_cover_what_the_trainer_logs():
    env = SpaceXEnv(level=0, fuel_budget=1e9)
    env.reset(seed=4)
    for _ in range(3000):
        _, _, term, trunc, info = env.step(IDLE)
        if term or trunc:
            break
    m = info["episode_metrics"]
    for k in ("level", "rules_stage", "success", "soft", "merit", "touchdown", "dist_m", "speed_mps", "tilt_rad", "fuel_used", "out_of_fuel", "end", "crashed",
              "tilt_out", "stall_out", "ignitions", "steps", "env_ok", "brake_peak", "ov_0", "ov_7", "term_crash_pen"):
        assert k in m


def test_stepping_is_fast():
    env = SpaceXEnv(level=0, fuel_budget=1e9)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    t0, n = time.time(), 0
    while n < 1500:
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 5).astype(np.float32))
        n += 1
        if term or trunc:
            env.reset()
    assert (time.time() - t0) / n < 0.004                                               # about 0.2 ms here; the old Box2D stack needed about 1 ms


def test_calibrated_vehicle_numbers():
    from src.sim.calibrate import engine_table
    t = engine_table(C.FUEL_BUDGET)
    assert t["can_hover"] is False                                                      # the minimum throttle out-pushes the weight
    assert 3.0 < t["sea level"]["twr_full"] < 3.6 and 1.2 < t["sea level"]["twr_min"] < 1.5
    assert t["sea level"]["net_brake_full_mps2"] == pytest.approx(C.BRAKE_MAX_MPS2, abs=2.5)   # the configured braking limit matches the engine table
    assert 160 < t["terminal_velocity_mps"]["0 m"] < 210


def test_the_scripted_expert_lands_softly_on_the_pad_from_the_easiest_spawns():
    rng = np.random.default_rng(2)
    good = 0
    for _ in range(16):
        w = Wind(0.0, rng)
        s, _ = make_spawn(rng, 0, prop=C.FUEL_BUDGET, wind=w)
        r = rollout(s, w)
        good += (r["t"] < 119 and r["speed"] <= C.SUCCESS_MAX_SPEED_MPS and r["tilt"] <= C.SUCCESS_MAX_TILT_RAD and abs(r["x"]) <= C.SUCCESS_MAX_DIST_M)
        assert 1500 < r["prop_used"] < 4500 or r["t"] >= 119
    assert good >= 14                                                                   # the expert lands softly, on the pad, almost every time at level 0


def test_the_spawn_is_aimed_at_a_normal_landing_not_at_the_passive_fall():
    """The burn lasts ~9 s longer than the fall, so the start is aimed where a straight burn plus a constant-rate sideways stop would end."""
    from src.sim.physics import powered_descent
    rng = np.random.default_rng(3)
    for level in (0, 1):
        for _ in range(4):
            s, d = make_spawn(rng, level, prop=C.FUEL_BUDGET)
            stop_x, burn_h, total = powered_descent(s)
            assert stop_x == pytest.approx(d["target_miss_m"], abs=1.0)
            assert 300 < burn_h < 1500 and total > 15


def test_entry_angle_presets_and_trajectory_variation(monkeypatch):
    from src.sim.spawn import entry_range
    ranges = {k: entry_range({"entry_deg": (1.0, 2.0)}) for k in ("extreme", "intermediate", "SpaceX", "straight down", "normal")}
    monkeypatch.setattr(C, "ENTRY_PRESET", "normal")
    assert entry_range({"entry_deg": (1.0, 2.0)}) == (1.0, 2.0)                                           # normal = the level's own range
    for name, (lo, hi) in (("extreme", (35, 48)), ("intermediate", (48, 62)), ("SpaceX", (62, 74)), ("straight down", (87, 90))):
        monkeypatch.setattr(C, "ENTRY_PRESET", name)
        assert entry_range({"entry_deg": (1.0, 2.0)}) == (lo, hi)
        angles = [make_spawn(np.random.default_rng(i), 0, prop=4500.0)[1]["entry_deg"] for i in range(12)]
        assert all(lo - 1e-9 <= a <= hi + 1e-9 for a in angles) and max(angles) - min(angles) > 0.3 * (hi - lo)
    monkeypatch.setattr(C, "ENTRY_PRESET", "SpaceX")
    monkeypatch.setattr(C, "TRAJ_VARIATION", 0.0)
    mid = [make_spawn(np.random.default_rng(i), 0, prop=4500.0)[1] for i in range(6)]
    assert all(abs(d["entry_deg"] - 68.0) < 1e-6 for d in mid)                                           # no variation: always the middle of the range
    assert len({round(d["speed"], 3) for d in mid}) == 1
    monkeypatch.setattr(C, "TRAJ_VARIATION", 0.5)
    half = [make_spawn(np.random.default_rng(i), 0, prop=4500.0)[1]["entry_deg"] for i in range(30)]
    assert all(65 - 1e-6 <= a <= 71 + 1e-6 for a in half) and max(half) - min(half) > 3.0                 # half the spread, around the middle


def test_the_default_start_is_unchanged_by_the_new_entry_settings():
    """normal + full variation draws exactly what the old code drew (same seed, same start)."""
    rng = np.random.default_rng(5)
    v = float(rng.uniform(C.SPACEX_V_MIN, C.SPACEX_V_MAX))
    g = float(rng.uniform(*C.LEVELS[0]["entry_deg"]))
    d = make_spawn(np.random.default_rng(5), 0, prop=4500.0)[1]
    assert d["entry_deg"] == pytest.approx(g, abs=1e-9) and d["speed"] == pytest.approx(v * C.LEVELS[0]["speed"], abs=1e-9)
