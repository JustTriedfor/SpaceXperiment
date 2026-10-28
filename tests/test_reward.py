"""The reward (real-scale constants): properties of the shaping, the scores and the events, tested against the module's own constants."""
import math

import numpy as np
import pytest

from src.sim.env import SpaceXEnv
from src.training import reward as R
from src.training.reward import State, compute_reward
from src.utils import config as C


def st(dist, speed, height, angle=0.0, spin=0.0, legs=0, vy=None, x=0.0, vx=0.0):
    return State(dist, speed, height, angle, spin, legs, -speed if vy is None else vy, x, vx)


def td(dist, speed, tilt):
    return dict(dist_m=dist, speed_mps=speed, tilt_rad=tilt)


# ---------------------------------------------------------------- shaping
def test_the_envelope_is_the_speed_from_which_the_planned_braking_still_stops():
    assert R.envelope_speed(0.0) == R.ENVELOPE_FLOOR
    h = 400.0
    assert R.envelope_speed(h) == pytest.approx(math.sqrt(2 * R.ENVELOPE_DECEL * h) + R.ENVELOPE_FLOOR)
    assert R.envelope_speed(h) < R.envelope_speed(4 * h) < 2.1 * R.envelope_speed(h)
    assert R.ENVELOPE_DECEL < C.BRAKE_MAX_MPS2                                  # the plan asks for less than the engine can give


def test_potential_rewards_closer_slower_and_upright():
    h = 500.0
    ok = R.potential(st(600.0, 0.9 * R.envelope_speed(h), h))
    assert R.potential(st(400.0, 0.9 * R.envelope_speed(h), h)) > ok                 # closer
    assert R.potential(st(600.0, 3.0 * R.envelope_speed(h), h)) < ok                 # far above the safe-descent envelope
    assert R.potential(st(600.0, 0.5 * R.envelope_speed(h), h)) == pytest.approx(ok)    # below the envelope: no further credit
    assert R.potential(st(600.0, 0.9 * R.envelope_speed(h), h, angle=0.3)) < ok      # tilted
    assert R.potential(st(600.0, 0.9 * R.envelope_speed(h), h, spin=0.2)) < ok       # spinning
    assert R.potential(st(600.0, 0.9 * R.envelope_speed(h), 0.0, legs=2)) > R.potential(st(600.0, 0.9 * R.envelope_speed(h), 0.0, legs=0))


def test_free_fall_nets_negative_shaping_at_these_speeds():
    a, b = st(2000.0, 250.0, 1800.0), st(1800.0, 262.0, 1600.0)                    # 200 m closer but faster
    p0, p1 = R.potential(a), R.potential(b)
    assert p1 - p0 < 150.0                                                          # the distance gain is modest next to the overspeed cost


# ---------------------------------------------------------------- the slow-approach charge
def test_slow_charge_formula_and_exemptions():
    h = R.SLOW_REF_HEIGHT + R.SLOW_GROWTH_SPAN
    rate = R.slow_rate_per_sec(st(100.0, 0.0, h))
    assert rate == pytest.approx(R.SLOW_RATE_REF * R.SLOW_GROWTH)                   # one growth span above the reference height, fully stopped
    assert R.slow_rate_per_sec(st(100.0, 0.0, R.SLOW_REF_HEIGHT)) == pytest.approx(R.SLOW_RATE_REF)
    assert rate > R.slow_rate_per_sec(st(100.0, 0.0, h - 100)) > R.slow_rate_per_sec(st(100.0, 0.0, h - 200))     # exponential in altitude
    assert R.slow_rate_per_sec(st(100.0, 0.0, R.SLOW_MIN_HEIGHT)) == 0.0            # the final approach is exempt
    assert R.slow_rate_per_sec(st(100.0, 0.0, h, legs=1)) == 0.0                    # on the ground
    assert R.slow_rate_per_sec(st(100.0, 2.0 * R.envelope_speed(h), h)) == 0.0      # fast enough: nothing to pay
    cap = R.slow_rate_per_sec(st(100.0, 0.0, R.SLOW_MAX_HEIGHT))
    assert R.slow_rate_per_sec(st(100.0, 0.0, R.SLOW_MAX_HEIGHT * 3)) == pytest.approx(cap)   # the exponential is capped


# ---------------------------------------------------------------- landing scores
def test_landing_tiers_and_the_perfect_score():
    zone = C.SUCCESS_MAX_DIST_M
    assert R.landing_score(td(zone * 3, 0.5, 0.0)) == R.LAND_OFF_PAD
    assert R.landing_score(td(zone * 0.5, R.SOFT_SPEED * 2, 0.0)) == R.LAND_ZONE            # survived in the zone but hard
    assert R.landing_score(td(zone, R.SOFT_SPEED, R.SOFT_TILT)) == pytest.approx(R.LAND_SOFT_EDGE)
    assert R.landing_score(td(0.0, 0.0, 0.0)) == pytest.approx(R.LAND_PERFECT)
    scores = [R.landing_score(td(d, 1.0, 0.02)) for d in np.linspace(0, zone, 11)]
    assert all(a > b for a, b in zip(scores, scores[1:]))                                      # closer to the centre always pays more
    t = R.landing_terms(td(1.0, 0.5, 0.01))
    assert t["land"] + t["soft"] == pytest.approx(R.landing_score(td(1.0, 0.5, 0.01))) and t["land"] == R.LAND_SOFT_EDGE


# ---------------------------------------------------------------- crash scores
def test_crash_position_inside_the_zone_rises_to_the_centre_and_fades_with_speed():
    zone = C.SUCCESS_MAX_DIST_M
    slow = lambda d: R.crash_position_score(dict(dist_m=d, speed_mps=R.CRASH_SPEED_GOAL))
    assert slow(zone) == pytest.approx(R.CRASH_IN_EDGE) and slow(0.0) == pytest.approx(R.CRASH_IN_CENTER)
    assert all(slow(a) > slow(b) for a, b in [(0, 3), (3, 6), (6, zone)])
    fast = R.crash_position_score(dict(dist_m=0.0, speed_mps=R.CRASH_SPEED_GOAL + R.CRASH_POS_FADE_MPS / 2))
    assert fast == pytest.approx(0.5 * R.CRASH_IN_CENTER)
    assert R.crash_position_score(dict(dist_m=0.0, speed_mps=300.0)) == 0.0                 # a fast crash in the middle earns no bonus


def test_crash_position_outside_the_zone_is_a_cliff_then_exponential_and_capped():
    zone = C.SUCCESS_MAX_DIST_M
    out = lambda d, v=300.0: R.crash_position_score(dict(dist_m=d, speed_mps=v))
    assert out(zone + 1e-6) == pytest.approx(R.CRASH_OUT_EDGE, rel=1e-3)
    assert out(zone + 50) == pytest.approx(R.CRASH_OUT_EDGE * R.CRASH_OUT_GROWTH ** 50) and out(zone + 50) < out(zone + 10)
    assert out(zone + 5000) == R.CRASH_OUT_CAP
    assert out(zone + 50, 300.0) == out(zone + 50, 1.0)                                      # penalties never fade with speed


def test_crash_speed_score_slope_and_independence():
    assert R.crash_speed_score(R.CRASH_SPEED_GOAL) == 0.0 and R.crash_speed_score(0.0) == 0.0
    assert R.crash_speed_score(R.CRASH_SPEED_GOAL + 10) == pytest.approx(10 * R.CRASH_SPEED_PER_MPS)
    a = R.crash_terms(td(5.0, 40.0, 0.0))
    b = R.crash_terms(td(5.0, 80.0, 0.0))
    assert b["crash_speed"] < a["crash_speed"] and b["crash_pos"] <= a["crash_pos"]
    assert R.crash_terms(None) == dict(crash_pos=0.0, crash_speed=0.0, crash_vert=0.0)


def test_a_fast_crash_is_deeply_negative_even_dead_centre():
    t = R.crash_terms(td(0.0, 200.0, 0.0))
    assert t["crash_pos"] + t["crash_speed"] < -3000


# ---------------------------------------------------------------- events in compute_reward
def test_events_pay_once_and_only_where_they_belong():
    s = st(1000.0, 200.0, 800.0)
    base = compute_reward(s, s, 0.0)[1]
    assert base["first_fire"] == 0 and base["refire"] == 0 and base["crash_pen"] == 0
    assert compute_reward(s, s, 0.0, ignition=1)[1]["first_fire"] == R.FIRST_IGNITION_BONUS
    assert compute_reward(s, s, 0.0, ignition=2, refire_on=True)[1]["refire"] == -R.REFIRE_BASE
    assert compute_reward(s, s, 0.0, ignition=4, refire_on=True)[1]["refire"] == -R.REFIRE_BASE * R.REFIRE_GROWTH ** 2
    for end in ("crash", "oob"):
        t = compute_reward(s, s, 0.0, end=end, arrival=td(5.0, 50.0, 0.1))[1]
        assert t["crash_pen"] == R.CRASH_PENALTY and t["crash_speed"] < 0
    assert compute_reward(s, s, 0.0, end="crash", arrival=None)[1]["crash_pen"] == R.CRASH_PENALTY
    for end in ("tilt", "stall", "timeout"):
        assert compute_reward(s, s, 0.0, end=end, arrival=td(5.0, 50.0, 0.1))[1]["crash_pen"] == 0
    assert compute_reward(s, s, 0.0, end="tilt", arrival=td(5.0, 50.0, 0.1))[1]["tilt_out"] == R.TILT_OUT_PENALTY
    assert compute_reward(s, s, 0.0, end="stall", arrival=td(5.0, 50.0, 0.1))[1]["stall_out"] == R.STALL_OUT_PENALTY
    assert compute_reward(s, s, 0.0, end="timeout")[1]["timeout"] == R.TIMEOUT_PENALTY
    assert compute_reward(s, s, 0.0, fuel_out_event=True)[1]["fuel_out"] == R.FUEL_OUT_PENALTY
    assert compute_reward(s, s, 0.0, ascend_event=True)[1]["ascend"] == R.ASCEND_PENALTY
    t = compute_reward(s, s, 0.0, end="rest", landing=td(1.0, 0.5, 0.01))[1]
    assert t["land"] + t["soft"] > R.LAND_ZONE and t["crash_pen"] == 0


def test_off_course_charge_follows_the_predicted_touchdown():
    on = st(100.0, 100.0, 500.0, x=0.0, vx=0.0)
    off = st(100.0, 100.0, 500.0, x=R.OFFCOURSE_TOLERANCE_M * 5, vx=0.0)
    assert compute_reward(on, on, 0.0)[1]["offcourse"] == 0.0
    assert compute_reward(off, off, 0.0)[1]["offcourse"] == pytest.approx(R.OFFCOURSE_PER_SEC / C.FPS)
    grounded = st(100.0, 1.0, 0.0, legs=2, x=R.OFFCOURSE_TOLERANCE_M * 5)
    assert compute_reward(grounded, grounded, 0.0)[1]["offcourse"] == 0.0


# ---------------------------------------------------------------- in the real environment
def zero_action():
    return np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def test_a_free_fall_crash_is_scored_by_the_environment():
    env = SpaceXEnv(level=0, fuel_budget=1e9)
    env.reset(seed=5)
    total = 0.0
    for _ in range(3000):
        _, r, term, trunc, info = env.step(zero_action())
        total += r
        if term or trunc:
            break
    m = info["episode_metrics"]
    assert m["end"] in ("crash", "oob") and m["merit"] == 0 and not m["success"] and not m["soft"]
    assert m["term_crash_pen"] == R.CRASH_PENALTY and m["term_land"] == 0 and m["term_first_fire"] == 0 and m["term_crash_speed"] < 0
    assert total * 1.0 < 0
    assert m["term_hover"] == 0 or m["term_hover"] < 0


def test_the_first_ignition_is_paid_once_and_the_ascent_once(monkeypatch):
    env = SpaceXEnv(level=0, fuel_budget=1e9)
    env.reset(seed=1)
    on = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    paid = [env.step(on)[4]["reward_terms"]["first_fire"] for _ in range(4)]
    assert paid[0] == R.FIRST_IGNITION_BONUS and sum(paid) == R.FIRST_IGNITION_BONUS
    env2 = SpaceXEnv(level=0, fuel_budget=1e9)
    env2.reset(seed=1)
    env2.s.vy = 40.0                                                                          # rising
    pen = [env2.step(zero_action())[4]["reward_terms"]["ascend"] for _ in range(3)]
    assert pen[0] == R.ASCEND_PENALTY and sum(pen) == R.ASCEND_PENALTY


def test_running_out_of_propellant_is_paid_once():
    env = SpaceXEnv(level=0, fuel_budget=300.0)                                              # one second of full thrust
    env.reset(seed=2)
    on = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    events, out = 0, False
    for _ in range(80):
        _, _, term, trunc, info = env.step(on)
        events += bool(info["fuel_out_event"])
        out = out or info["out_of_fuel"]
        if term or trunc:
            break
    assert events == 1 and out
