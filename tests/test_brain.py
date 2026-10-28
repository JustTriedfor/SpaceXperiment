import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from src.app import server
from src.training.brain import episode_payload, network_trace, policy_weights
from src.training.env import make_vec_env
from src.utils import config as C


@pytest.fixture(scope="module")
def model():
    vec = make_vec_env(1, 0, level=0, reward="custom")                    # one env: in-process, no worker processes
    return PPO("MlpPolicy", vec, device="cpu", verbose=0, seed=0, policy_kwargs=dict(net_arch=dict(pi=list(C.NET_ARCH), vf=list(C.NET_ARCH))))


def test_trace_matches_the_policys_own_outputs(model):
    obs = np.random.default_rng(1).normal(size=(7, 18)).astype(np.float32)
    acts = np.zeros((7, 5), dtype=np.float32)
    t = network_trace(model.policy, obs, acts)
    assert t["dims"] == [18, 256, 256, 5]
    mean = np.array(t["m"]).reshape(7, 5)
    predicted, _ = model.policy.predict(obs, deterministic=True)
    assert np.allclose(predicted, np.clip(mean, -1, 1), atol=2e-3)         # the deterministic action is the clipped mean
    values = model.policy.predict_values(torch.as_tensor(obs)).detach().numpy().ravel()
    assert np.allclose(values, t["v"], atol=0.01)


def test_hidden_activations_are_tanh_bounded_and_flat_row_major(model):
    obs = np.random.default_rng(2).normal(size=(5, 18)).astype(np.float32) * 3
    t = network_trace(model.policy, obs)
    h1, h2 = np.array(t["h1"]).reshape(5, 256), np.array(t["h2"]).reshape(5, 256)
    assert np.abs(h1).max() <= 100 and np.abs(h2).max() <= 100            # activations x100 as ints, tanh range
    pn = model.policy.mlp_extractor.policy_net
    with torch.no_grad():
        ref = torch.tanh(pn[0](torch.as_tensor(obs))).numpy()
    assert np.allclose(h1, np.round(ref * 100), atol=1)                   # step t is row t
    assert "a" not in t                                                    # no actions given, none returned


def test_weights_have_the_expected_shapes_and_the_policy_stays_in_its_mode(model):
    w = policy_weights(model.policy)
    assert (len(w["w1"]), len(w["w2"]), len(w["w3"])) == (256 * 18, 256 * 256, 5 * 256)
    model.policy.set_training_mode(True)
    episode_payload(model.policy, np.zeros((2, 18), np.float32), np.zeros((2, 5), np.float32))
    assert model.policy.training is True                                   # tracing must not leave the policy in eval mode


def test_the_server_sends_activations_only_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "LOG_DIR", tmp_path)
    live = tmp_path / "myrun" / "live"
    live.mkdir(parents=True)
    for i in (1, 2):
        (live / f"ep_{i:08d}_e3.json").write_text(json.dumps(
            {"id": i, "env": 3, "timesteps": 100 * i, "frames": [], "act": {"x": [0.0]}, "level": 0}))
    ep = server.live_episode("myrun", 3, 0)
    assert ep["id"] == 2 and "act" not in ep                               # newest episode, without the 160 KB payload
    assert "act" in server.live_episode("myrun", 3, 0, want_act=True)
    old = server.live_episode("myrun", 3, 0, want_act=True, ep_id=1)       # fetch a specific episode (used when you pick a colour)
    assert old["id"] == 1 and old["act"] == {"x": [0.0]}
    assert server.live_episode("myrun", 3, 0, want_act=True, ep_id=99).get("wait") is True


def test_trainer_records_activations_that_line_up_with_the_frames():
    tag = "_braintest"
    name = f"custom_s0{tag}"
    overrides = {"config": {"FUEL_BUDGET": 100000.0}}
    try:
        out = subprocess.run(
            [sys.executable, "src/training/train.py", "--reward", "custom", "--seed", "0", "--total-steps", "9000",
             "--n-envs", "2", "--tag", tag],
            cwd=C.ROOT, capture_output=True, text=True, timeout=300, env=dict(os.environ, ROCKETEER_OVERRIDES=json.dumps(overrides)))
        assert out.returncode == 0, out.stderr[-600:]
        assert "[brain] activations skipped" not in out.stdout
        files = sorted((C.LOG_DIR / name / "live").glob("ep_*.json"))
        assert files
        checked = 0
        for f in files:
            ep = json.loads(f.read_text())
            act, T = ep["act"], len(ep["frames"])
            assert act["dims"] == [18, 256, 256, 5]
            # one input row, one hidden row, one action per step: exactly the frames of the episode
            assert len(act["x"]) == T * 18 and len(act["h1"]) == T * 256 and len(act["h2"]) == T * 256
            assert len(act["m"]) == T * 5 and len(act["a"]) == T * 5 and len(act["v"]) == T
            assert (len(act["w1"]), len(act["w2"]), len(act["w3"])) == (256 * 18, 256 * 256, 5 * 256)
            x = np.array(act["x"]).reshape(T, 18)
            assert x[0, 6] == 1.0 and x[0, 7] == 0.0 and x[0, 11] == 0.0                # first input: full tank, thrust level zero, engine off
            assert abs(x[0, 0] * C.OBS_X_TO_M - ep["frames"][0]["x"]) < 40.0                # the first input is the state at the first frame (one step of travel apart)
            assert abs(x[-1, 0] * C.OBS_X_TO_M - ep["frames"][-1]["x"]) < 40.0
            checked += 1
        assert checked >= 3
    finally:
        shutil.rmtree(C.LOG_DIR / name, ignore_errors=True)
        shutil.rmtree(C.DATA_DIR / "trajectories" / name, ignore_errors=True)
        for p in list(C.MODEL_CONFIG_DIR.glob(f"{name}.json")) + list(C.CHECKPOINT_DIR.glob(f"{name}*")):
            p.unlink(missing_ok=True)
        (C.LOG_DIR / f"{name}.out").unlink(missing_ok=True)


def test_frames_carry_the_wind_and_the_actuator_state():
    from src.sim.env import SpaceXEnv
    for level, windy in ((0, False), (2, True)):
        env = SpaceXEnv(level=level, record=True, fuel_budget=1e9)
        env.reset(seed=3)
        for _ in range(200):
            _, _, term, trunc, _ = env.step(np.array([1.0, 0.5, -0.5, 0.3, -0.3], np.float32))
            if term or trunc:
                break
        winds = [f["wd"] for f in env.frames]
        assert (max(abs(w) for w in winds) > 0.1) == windy and max(abs(w) for w in winds) <= C.LEVELS[level]["wind_power"] + 1e-9
        f = env.frames[min(60, len(env.frames) - 1)]
        for k in ("g", "f", "r1", "r2", "q", "mass", "thr", "gas", "vx", "vy", "ax", "ay", "dr", "on"):
            assert k in f                                                                      # what the viewer's telemetry and animations read
        assert f["on"] == 1 and f["m"] > 0.3 and abs(f["g"]) > 0 and abs(f["f"]) > 0 and f["r1"] != 0 and f["r2"] != 0



def test_payload_lets_the_browser_run_the_network_itself(model):
    """The decision-boundary view evaluates the policy in JavaScript from the shipped weights and biases: replicate it here."""
    from src.training.brain import episode_payload
    obs = np.random.default_rng(5).normal(size=(6, 18)).astype(np.float32)
    p = episode_payload(model.policy, obs, np.zeros((6, 5), np.float32))
    assert {"b1", "b2", "b3", "scales"} <= set(p) and set(p["scales"]) == {"x", "y", "vx", "vy"}
    w1, w2, w3 = (np.array(p[k]).reshape(s) for k, s in (("w1", (256, 18)), ("w2", (256, 256)), ("w3", (5, 256))))
    for t in range(6):
        h1 = np.tanh(w1 @ obs[t] + np.array(p["b1"]))
        h2 = np.tanh(w2 @ h1 + np.array(p["b2"]))
        mean = w3 @ h2 + np.array(p["b3"])
        assert np.allclose(mean, np.array(p["m"]).reshape(6, 5)[t], atol=0.02)
