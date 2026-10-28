"""Two neural networks in one environment (start profile "pair")."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.app import jobs
from src.sim.dual_env import DualEnv, agent_view
from src.sim.env import SpaceXEnv
from src.training import data as D
from src.training import pair as P
from src.training import weights_io as W
from src.utils import config as C


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    for name, sub in (("LOG_DIR", "logs"), ("CHECKPOINT_DIR", "ckpt"), ("MODEL_CONFIG_DIR", "cfg"), ("DATA_DIR", "data")):
        monkeypatch.setattr(C, name, tmp_path / sub)
        (tmp_path / sub).mkdir()
    monkeypatch.setattr(D, "ROOT_DIR", tmp_path / "data" / "trajectories")
    monkeypatch.setattr(jobs, "TRASH_DIR", tmp_path / "trash")
    W._INFO_CACHE.clear()
    return tmp_path


@pytest.fixture
def single(sandbox):
    m = PPO("MlpPolicy", DummyVecEnv([lambda: SpaceXEnv()]), seed=3, device="cpu", policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))
    m.save(str(C.CHECKPOINT_DIR / "single_a"))
    return m


def test_each_lander_gets_its_own_view_of_the_dual_observation():
    env = DualEnv(level=0)
    obs, _ = env.reset(seed=1)
    obs, r, *_ = env.step(np.zeros(10))
    a, b = agent_view(obs, 0), agent_view(obs, 1)
    assert a.shape == b.shape == (40,)
    assert np.allclose(a[:18], obs[:18]) and np.allclose(a[18:36], obs[18:36]) and np.allclose(b[:18], obs[18:36]) and np.allclose(b[18:36], obs[:18])    # own first, partner second
    assert a[36] == -b[36] and a[37] == -b[37] and (a[38], a[39]) == (b[39], b[38])                          # the gaps flip sign, the down flags swap
    assert agent_view(np.stack([obs, obs]), 1).shape == (2, 40)


def test_every_lander_has_its_own_reward_and_both_share_the_landing_together_terms():
    from src.sim.expert import Expert
    from src.sim.expert_data import command_to_action
    env = DualEnv(level=0)
    env.set_rules_stage(2)
    env.reset(seed=7)
    eb = Expert()
    eb.reset()
    ret = np.zeros(2)
    done = False
    while not done:
        sb = env.b.s
        act = np.concatenate([[-1.0, 0, 0, 0, 0], command_to_action(eb.act(sb, env.b.wind.at(sb.t, sb.y)))])      # A never lights its engine and crashes; B lands
        _, total, te, tr, info = env.step(act)
        ret += info["agent_rewards"]
        done = te or tr
    assert env.end == ["crash", "rest"]
    assert ret[0] < 0 < ret[1]                                                       # each network is judged on its own lander
    assert info["episode_metrics"]["term_sync_bonus"] == 0.0                         # no landing-together bonus when one is lost (so none is shared)


def test_a_pair_starts_as_two_copies_of_the_single_network_and_flies_like_it(sandbox, single):
    r = P.make_pair("single_a", "twin")
    assert r["files"] == ["twin_A.zip", "twin_B.zip"] and P.resolve_pair("twin")[0] == "twin" and P.resolve_pair("twin_B")[0] == "twin"
    with pytest.raises(ValueError, match="already exists"):
        P.make_pair("single_a", "twin")
    ia, ib = W.checkpoint_info("twin_A"), W.checkpoint_info("twin_B")
    assert ia["kind"] == ib["kind"] == "agent" and (ia["n_obs"], ia["n_act"]) == (40, 5) and ia["arch"] == [256, 256]
    mA, mB = P.load_pair_models("twin")
    rng = np.random.default_rng(0)
    own = rng.normal(size=(8, 18)).astype(np.float32)
    for m in (mA, mB):
        full = np.concatenate([own, rng.normal(size=(8, 22)).astype(np.float32) * 5], axis=1)          # the partner part is ignored at the start
        with torch.no_grad():
            got = m.policy.get_distribution(torch.as_tensor(full)).distribution.mean.numpy()
            want = single.policy.get_distribution(torch.as_tensor(own)).distribution.mean.numpy()
            v_got, v_want = m.policy.predict_values(torch.as_tensor(full)).numpy(), single.policy.predict_values(torch.as_tensor(own)).numpy()
        assert np.allclose(got, want, atol=1e-5) and np.allclose(v_got, v_want, atol=1e-4)


def test_the_two_networks_train_together_on_one_environment_and_stay_separate(sandbox, single, monkeypatch):
    from src.training.train import CurriculumCallback
    monkeypatch.setattr(C, "START_PROFILE", "pair")
    P.make_pair("single_a", "start")
    before = {k: m.policy.state_dict()["action_net.weight"].clone() for k, m in zip("AB", P.load_pair_models("start"))}
    args = SimpleNamespace(n_envs=1, seed=0, no_live=False, start_level=0, reward="custom", init_from="start", total_steps=1100, stop_at_level=None,
                           snapshot_every=0, live_dir=None, live_env=None, tag="")
    P.run_pair(args, "run1", C.LOG_DIR / "run1", dict(C.PPO_KWARGS, learning_rate=1e-3), CurriculumCallback)
    mA, mB = P.load_pair_models("run1")
    wa, wb = mA.policy.state_dict()["action_net.weight"], mB.policy.state_dict()["action_net.weight"]
    assert not torch.allclose(wa, before["A"]) and not torch.allclose(wb, before["B"])                   # both learned
    assert not torch.allclose(wa, wb)                                                                    # ... differently: they are separate networks
    assert (C.LOG_DIR / "run1" / "episodes.csv").exists() and (C.MODEL_CONFIG_DIR / "run1.json").exists()
    assert json.loads((C.MODEL_CONFIG_DIR / "run1.json").read_text())["networks"] == 2
    ep = json.loads(sorted((C.LOG_DIR / "run1" / "live").glob("ep_*.json"))[0].read_text())
    assert ep["dual"]["spacing"] == C.DUAL_PAD_SPACING_M and ep["act"]["dims"][0] == 40 and ep["actB"]["dims"][-1] == 5        # both networks' activations are streamed


def test_a_pair_is_evaluated_listed_quieted_and_trashed_as_one_thing(sandbox, single):
    from src.sim.evaluate import evaluate
    jobs.ckpt_make_pair(dict(**{"from": "single_a"}, out="duo"))
    res = evaluate("duo", [0], 2)
    assert res["kind"] == "pair" and res["checkpoint"] == "duo" and "gap_median" in res["levels"][0]
    rows = {r["name"]: r for r in jobs.checkpoint_details("du")}
    assert "duo_A" not in rows and "duo_B" not in rows and rows["duo"]["kind"] == "pair" and rows["duo"]["files"] == ["duo_A", "duo_B"] and rows["duo"]["busy"]
    jobs.ckpt_quiet(dict(**{"from": "duo"}, out="duo_quiet", std=0.05))
    assert W.checkpoint_info("duo_quiet_A")["std"] == pytest.approx(0.05, abs=0.002) and W.checkpoint_info("duo_quiet_B")["std"] == pytest.approx(0.05, abs=0.002)
    with pytest.raises(ValueError, match="already exists"):
        jobs.ckpt_quiet(dict(**{"from": "duo"}, out="duo_quiet", std=0.05))
    jobs.ckpt_delete(dict(name="duo"))
    assert not (C.CHECKPOINT_DIR / "duo_A.zip").exists() and not (C.CHECKPOINT_DIR / "duo_B.zip").exists() and len(list((sandbox / "trash").rglob("duo_*.zip"))) == 2


def test_the_pair_profile_is_offered_by_the_app_and_a_pair_may_start_a_run():
    from src.app.schema import build_run
    assert "pair" in {f["key"]: f for f in __import__("src.app.schema", fromlist=["public_schema"]).public_schema()}["start_profile"]["options"]
    r = build_run({"start_profile": "pair"})
    assert "errors" not in r and r["overrides"]["config"]["START_PROFILE"] == "pair"
    assert "errors" in build_run({"start_profile": "pair", "init_from": "no_such_pair_anywhere"})
