"""Staged rules: stage 0 (tilt, slow charge, crash/landing scores), stage 1 (+ clock and stall), stage 2 (+ ignition rules)."""
import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from src.training.curriculum import RulesGate
from src.utils import config as C


# ============================================================== the gate (pure logic)
def gate(**kw):
    args = dict(stage=0, max_stage=2, window=10, rate=0.2, max_steps=0)
    args.update(kw)
    return RulesGate(**args)


def feed(g, outcomes, stage=None):
    for o in outcomes:
        g.observe(g.stage if stage is None else stage, o)


def test_defaults_mean_no_staging():
    assert C.RULES_STAGE_START == C.RULES_STAGE_MAX == 2                   # every rule on unless the app turns staging on
    assert C.RULES_GATE_RATE == 0.20 and C.RULES_GATE_WINDOW == 200


def test_gate_waits_for_a_full_window():
    g = gate()
    feed(g, [True] * 9)                                                    # 100% survived, but only 9 of 10 episodes
    assert g.check(1000) is None
    feed(g, [True])
    assert g.check(2000) == (1, "survival", 1.0)


def test_gate_needs_the_survival_rate():
    g = gate(rate=0.3)
    feed(g, [True] * 2 + [False] * 8)                                      # 20% < 30%
    assert g.check(1000) is None
    feed(g, [True] * 3 + [False] * 7)                                      # the window slides: now 30%
    assert g.check(2000)[0] == 1


def test_episodes_from_an_older_stage_are_ignored():
    g = gate(stage=1)
    feed(g, [True] * 50, stage=0)                                          # still running under the old rules
    assert len(g.results) == 0 and g.check(1000) is None


def test_gate_advances_one_stage_at_a_time_and_resets_its_window():
    g = gate()
    feed(g, [True] * 10)
    assert g.check(500)[0] == 1 and g.stage == 1 and len(g.results) == 0 and g.stage_start == 500
    assert g.check(600) is None                                            # a fresh window is needed for the next stage
    feed(g, [True] * 10)
    assert g.check(900)[0] == 2
    feed(g, [True] * 10)
    assert g.check(1200) is None and g.stage == 2                          # nothing beyond the last stage


def test_gate_max_steps_forces_the_advance():
    g = gate(max_steps=1000)
    feed(g, [False] * 10)
    assert g.check(999) is None
    assert g.check(1000) == (1, "max_steps", 0.0)
    feed(g, [False] * 10)                                                   # a new stage starts with an empty window
    assert g.check(1999) is None and g.check(2000)[1] == "max_steps"       # counted from the start of the stage


def test_gate_max_steps_zero_never_forces():
    g = gate(max_steps=0)
    feed(g, [False] * 10)
    assert g.check(10_000_000) is None


# ============================================================== each rule follows the stage (real environment)
def staged_env(monkeypatch, stage):
    from src.sim.env import SpaceXEnv
    monkeypatch.setattr(C, "RULES_STAGE_START", stage)
    env = SpaceXEnv(level=0, fuel_budget=1e9)
    env.reset(seed=3)
    return env


IDLE = np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)          # engine off, everything centred


def test_the_clock_is_off_at_stage_0_and_on_from_stage_1(monkeypatch):
    for stage, expect in [(0, False), (1, True), (2, True)]:
        env = staged_env(monkeypatch, stage)
        env.deadline = 3                                                  # a clock of 3 steps
        end = None
        for _ in range(6):
            _, _, term, trunc, info = env.step(IDLE)
            if term or trunc:
                end = info["episode_metrics"]["end"]
                break
        assert (end == "timeout") is expect


def test_the_stall_rule_follows_the_stage(monkeypatch):
    monkeypatch.setattr(C, "STALL_GRACE_STEPS", 0)
    for stage, stalled in [(0, False), (1, True), (2, True)]:
        env = staged_env(monkeypatch, stage)
        env.s.vx = env.s.vy = 0.0                                          # hanging in the air at 3 km
        _, _, term, _, info = env.step(IDLE)
        assert bool(info.get("stall_out")) is stalled


def test_the_ignition_limit_is_enforced_only_at_stage_2_but_counting_always_runs(monkeypatch):
    monkeypatch.setattr(C, "MAX_IGNITIONS", 2)
    on = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for stage, blocked_expected in [(0, False), (1, False), (2, True)]:
        env = staged_env(monkeypatch, stage)
        blocked = False
        for a in [on, IDLE] * 4:
            for _ in range(3):
                _, _, _, _, info = env.step(a)
            blocked = blocked or info["ignition_blocked"]
        assert blocked is blocked_expected
        assert info["ignitions"] == (2 if stage == 2 else 4)


def test_the_refire_penalty_only_at_stage_2_but_the_first_ignition_bonus_always():
    from src.training import reward as R
    s = R.State(1000.0, 300.0, 1000.0, 0.0, 0.0, 0, -300.0)
    assert R.compute_reward(s, s, 0.0, ignition=3, refire_on=False)[1]["refire"] == 0.0
    assert R.compute_reward(s, s, 0.0, ignition=3, refire_on=True)[1]["refire"] == -50.0
    assert R.compute_reward(s, s, 0.0, ignition=1, refire_on=False)[1]["first_fire"] == 500.0


# ============================================================== performance gates (a cap alone never advances a failing agent)
def test_the_step_cap_holds_a_stage_when_nobody_survives_and_advances_one_that_does():
    from src.training.curriculum import RulesGate
    g = RulesGate(0, 2, 10, 0.5, max_steps=1000, cap_min_rate=0.10)
    feed(g, [False] * 10)
    assert g.check(5000) is None and g.stage == 0                         # past the cap but survival 0%: hold
    feed(g, [True] + [False] * 9)                                         # 10% survive
    assert g.check(5001)[1:] == ("max_steps", 0.1)                         # now the cap is a fair fallback
    feed(g, [False] * 9)                                                  # a partly filled window never triggers the cap
    assert g.check(9000) is None


def test_a_cooldown_blocks_the_gate():
    g = gate(rate=0.2)
    feed(g, [True] * 10)
    assert g.check(100, blocked=True) is None and g.stage == 0
    assert g.check(100)[0] == 1


def test_level_gate_reasons():
    from src.training.curriculum import level_reason
    args = dict(advance_rate=0.8, max_steps=500_000, cap_min_rate=0.10)
    assert level_reason(True, 0.85, 1000, **args) == "success"
    assert level_reason(True, 0.01, 600_000, **args) is None                # the cap with a 1% rate: hold, do not advance
    assert level_reason(True, 0.12, 600_000, **args) == "max_steps"
    assert level_reason(False, 0.9, 600_000, **args) is None                # an incomplete window never decides
    assert level_reason(True, 0.5, 10_000, **args) is None


def test_the_clock_scales_with_the_spawn(monkeypatch):
    from src.sim.physics import min_landing_time
    assert min_landing_time(0.0, 5.0, 22.0) == 0.0
    t = min_landing_time(3000.0, 300.0, 22.0, 9.81, 3.0)
    assert 14 < t < 20                                                       # a short coast, then about 14 s of braking from ~330 m/s at 22 m/s^2
    assert min_landing_time(1500.0, 0.0, 22.0) < min_landing_time(3000.0, 0.0, 22.0)
    assert min_landing_time(5.0, 80.0, 22.0, 9.81, 3.0) == pytest.approx((80.0 - 3.0) / 22.0, rel=0.1)   # already too fast: brake at once
    env = staged_env(monkeypatch, 1)
    fastest = min_landing_time(env.s.foot_height(), -env.s.vy, C.BRAKE_MAX_MPS2, C.GRAVITY_MPS2, C.SUCCESS_MAX_SPEED_MPS)
    assert env.limit_s == pytest.approx(max(C.TIME_LIMIT_S, C.TIME_LIMIT_SCALE * fastest + C.TIME_LIMIT_PAD_S))
    assert env.deadline == round(env.limit_s * C.FPS) and env.limit_s > fastest


def test_trainer_walks_through_all_stages_and_logs_them():
    tag = "_stagetest"
    name = f"custom_s0{tag}"
    overrides = {"config": {"RULES_STAGE_START": 0, "RULES_GATE_WINDOW": 8, "RULES_GATE_RATE": 0.0, "GATE_COOLDOWN_STEPS": 0, "FUEL_BUDGET": 100000.0}}
    try:
        out = subprocess.run([sys.executable, "src/training/train.py", "--reward", "custom", "--seed", "0", "--total-steps", "12000", "--n-envs", "2",
                              "--tag", tag, "--no-live"], cwd=C.ROOT, capture_output=True, text=True, timeout=300,
                             env=dict(os.environ, ROCKETEER_OVERRIDES=json.dumps(overrides)))
        assert out.returncode == 0, out.stderr[-600:]
        stages = list(csv.DictReader(open(C.LOG_DIR / name / "stages.csv")))
        assert [int(r["stage"]) for r in stages] == [0, 1, 2]                 # advanced one at a time, in order
        assert stages[1]["reason"] == "survival" and stages[2]["reason"] == "survival"
        eps = list(csv.DictReader(open(C.LOG_DIR / name / "episodes.csv")))
        assert {int(e["rules_stage"]) for e in eps} == {0, 1, 2}
    finally:
        shutil.rmtree(C.LOG_DIR / name, ignore_errors=True)
        shutil.rmtree(C.DATA_DIR / "trajectories" / name, ignore_errors=True)
        for p in list(C.MODEL_CONFIG_DIR.glob(f"{name}.json")) + list(C.CHECKPOINT_DIR.glob(f"{name}*")):
            p.unlink(missing_ok=True)
        (C.LOG_DIR / f"{name}.out").unlink(missing_ok=True)
