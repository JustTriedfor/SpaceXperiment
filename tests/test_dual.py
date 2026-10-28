"""Dual land: one environment, two landers, two pads, landing together."""
import math

import numpy as np
import pytest
import torch

from src.sim.dual_env import DualEnv
from src.sim.expert import Expert
from src.sim.expert_data import command_to_action
from src.utils import config as C


def expert_actions(env, ea, eb):
    sa, sb = env.a.s, env.b.s
    return np.concatenate([command_to_action(ea.act(sa, env.a.wind.at(sa.t, sa.y))), command_to_action(eb.act(sb, env.b.wind.at(sb.t, sb.y)))])


def fly_pair(env, seed, skip_a=False):
    obs, info = env.reset(seed=seed)
    ea, eb = Expert(), Expert()
    ea.reset()
    eb.reset()
    done, n = False, 0
    while not done and n < 3000:
        a = expert_actions(env, ea, eb)
        if skip_a:
            a[:5] = [-1.0, 0, 0, 0, 0]                  # lander A never lights its engine
        obs, r, te, tr, info = env.step(a)
        done, n = te or tr, n + 1
    return info


def test_spaces_and_the_shared_predetermined_start():
    env = DualEnv(level=0)
    obs, info = env.reset(seed=3)
    assert env.observation_space.shape == (C.DUAL_OBS_DIM,) == (40,) and env.action_space.shape == (10,) == (C.DUAL_ACT_DIM,)
    assert obs.shape == (40,) and obs.dtype == np.float32 and info["spawn"]["spacing"] == C.DUAL_PAD_SPACING_M
    a, b = env.a.s, env.b.s
    assert a.foot_height() == pytest.approx(C.DUAL_SPAWN_HEIGHT_M, abs=0.2) and b.foot_height() == pytest.approx(C.DUAL_SPAWN_HEIGHT_M, abs=0.2)
    assert abs(a.vx - b.vx) < 0.15 * abs(a.vx) and abs(a.vy - b.vy) < 0.15 * abs(a.vy)             # the same path, give or take the small jitter
    assert abs(a.x - b.x) < 60.0                                                                       # ... in pad-relative coordinates
    assert env.a.wind is env.b.wind                                                                    # one shared wind


def test_without_jitter_the_two_starts_are_identical(monkeypatch):
    monkeypatch.setattr(C, "DUAL_JITTER", 0.0)
    env = DualEnv(level=1)
    env.reset(seed=5)
    a, b = env.a.s, env.b.s
    assert (a.x, a.y, a.vx, a.vy, a.th, a.om) == (b.x, b.y, b.vx, b.vy, b.th, b.om)


def test_two_experts_land_together_on_their_own_pads():
    env = DualEnv(level=0, record=True)
    env.set_rules_stage(2)
    good = 0
    for seed in range(6):
        info = fly_pair(env, 100 + seed)
        m = info["episode_metrics"]
        good += bool(m["success"]) and m["sync_dt_s"] <= C.DUAL_SYNC_TOL_S
    assert good >= 5
    d = info["trajectory"]["dual"]
    assert d["pads"] == [-C.DUAL_PAD_SPACING_M / 2, C.DUAL_PAD_SPACING_M / 2] and d["sync_dt"] is not None


def test_the_episode_waits_for_the_slower_lander_and_frames_are_padded():
    env = DualEnv(level=0, record=True)
    env.set_rules_stage(2)
    info = fly_pair(env, 7, skip_a=True)                      # A falls with its engine off and crashes; B lands
    m = info["episode_metrics"]
    tr = info["trajectory"]
    assert m["end"] == "crash" and not m["success"] and m["sync_dt_s"] != m["sync_dt_s"]             # no landing-together value when one is lost
    assert env.end == ["crash", "rest"] and env.t_end[0] < env.t_end[1]                              # A was down long before B ...
    assert len(tr["frames"]) == env.b.t                                                          # ... but the episode ran until B was down too
    last = tr["frames"][-1]
    assert last["m"] == 0.0 and last["vy"] == 0.0 and last["y"] == tr["frames"][env.t_end[0] - 1]["y"]       # A's frame is held in place after it is down


def test_frames_carry_both_landers_at_their_own_pads_and_the_spacing_is_adjustable(monkeypatch):
    monkeypatch.setattr(C, "DUAL_PAD_SPACING_M", 600.0)
    env = DualEnv(level=0, record=True)
    env.set_rules_stage(2)
    info = fly_pair(env, 11)
    tr = info["trajectory"]
    f = tr["frames"][-1]
    assert abs(f["x"] - (-300.0)) < 20 and abs(f["b"]["x"] - 300.0) < 20 and "b" not in f["b"]
    assert tr["dual"]["spacing"] == 600.0 and tr["pad"][0] == pytest.approx(-C.PAD_HALF_WIDTH_M - 300.0)
    assert len(f["legs"]) == 2 and len(f["b"]["legs"]) == 2


def test_the_landing_together_bonus_only_pays_when_both_survive_close_in_time():
    env = DualEnv(level=0)
    env.set_rules_stage(2)
    fly_pair(env, 21)
    assert env.sync_terminal > 0.0                                                                    # both landed within the tolerance
    fly_pair(env, 22, skip_a=True)
    assert env.sync_terminal == 0.0


def test_the_dual_network_starts_as_two_independent_single_networks():
    from stable_baselines3 import PPO
    from src.sim.dual_init import block_copy, new_dual_model
    from src.sim.env import SpaceXEnv
    from stable_baselines3.common.vec_env import DummyVecEnv
    single = PPO("MlpPolicy", DummyVecEnv([lambda: SpaceXEnv()]), device="cpu", seed=1, policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))
    dual = new_dual_model(seed=2)
    block_copy(single, dual)
    rng = np.random.default_rng(0)
    oa, ob = rng.normal(size=(6, 18)).astype(np.float32), rng.normal(size=(6, 18)).astype(np.float32)
    od = np.concatenate([oa, ob, rng.normal(size=(6, 4)).astype(np.float32)], axis=1)
    with torch.no_grad():
        mean_d = dual.policy.get_distribution(torch.as_tensor(od)).distribution.mean.numpy()
        mean_a = single.policy.get_distribution(torch.as_tensor(oa)).distribution.mean.numpy()
        mean_b = single.policy.get_distribution(torch.as_tensor(ob)).distribution.mean.numpy()
        v_d = dual.policy.predict_values(torch.as_tensor(od)).numpy()
        v_ab = single.policy.predict_values(torch.as_tensor(oa)).numpy() + single.policy.predict_values(torch.as_tensor(ob)).numpy()
    assert np.allclose(mean_d[:, :5], mean_a, atol=1e-5) and np.allclose(mean_d[:, 5:], mean_b, atol=1e-5)
    assert np.allclose(v_d, v_ab, atol=1e-4)                                                          # the critic is the sum of the two single critics


def test_the_profile_switch_builds_the_dual_environment(monkeypatch):
    from src.training import env as tenv
    monkeypatch.setattr(C, "START_PROFILE", "dual")
    e = tenv.make_env(level=0)
    assert isinstance(e, DualEnv)
    assert C.net_arch() == C.NET_ARCH_DUAL
    monkeypatch.setattr(C, "START_PROFILE", "spacex")
    assert C.net_arch() == C.NET_ARCH and not isinstance(tenv.make_env(level=0), DualEnv)


def test_the_app_exposes_the_dual_profile_and_its_parameters():
    from src.app.schema import build_run
    r = build_run({"start_profile": "dual", "dual_spacing": 1500, "dual_height": 5000})
    assert r["overrides"]["config"]["START_PROFILE"] == "dual"
    assert r["overrides"]["config"]["DUAL_PAD_SPACING_M"] == 1500.0 and r["overrides"]["config"]["DUAL_SPAWN_HEIGHT_M"] == 5000.0
    assert "errors" in build_run({"start_profile": "dual", "dual_spacing": 5})                         # below the minimum spacing


def test_pads_may_be_as_close_as_85_m_and_no_closer():
    from src.app.schema import build_run
    assert "errors" not in build_run({"start_profile": "dual", "dual_spacing": 85})
    assert "errors" in build_run({"start_profile": "dual", "dual_spacing": 80})
