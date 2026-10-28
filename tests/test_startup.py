"""The Startup tab's backend: a head start (copy a brain or run the expert pipeline) followed, if wanted, by training."""
import json
import time

import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.app import jobs, startup
from src.sim.env import SpaceXEnv
from src.training import weights_io as W
from src.utils import config as C


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    for name, sub in (("LOG_DIR", "logs"), ("CHECKPOINT_DIR", "ckpt"), ("MODEL_CONFIG_DIR", "cfg"), ("DATA_DIR", "data")):
        monkeypatch.setattr(C, name, tmp_path / sub)
        (tmp_path / sub).mkdir()
    monkeypatch.setattr(jobs, "PIPE_DIR", tmp_path / "logs" / "pipeline")
    monkeypatch.setattr(jobs, "TRASH_DIR", tmp_path / "trash")
    W._INFO_CACHE.clear()
    jobs.PIPE.update(proc=None, name=None)
    calls = dict(start=[], stop=0, running=False)
    monkeypatch.setitem(startup.HOOKS, "start", lambda p: (calls["start"].append(p), (200, dict(name="custom_s0_" + p["start_profile"])))[1])
    monkeypatch.setitem(startup.HOOKS, "stop", lambda: calls.__setitem__("stop", calls["stop"] + 1) or dict(stopped=[]))
    monkeypatch.setitem(startup.HOOKS, "status", lambda: dict(running=calls["running"], run=calls["start"][-1] and "custom_s0_" + calls["start"][-1]["start_profile"] if calls["start"] else None,
                                                             timesteps=1000, total_steps=2000, level=0))
    sandbox_calls = calls
    return tmp_path, sandbox_calls


@pytest.fixture
def single(sandbox):
    m = PPO("MlpPolicy", DummyVecEnv([lambda: SpaceXEnv()]), seed=1, device="cpu", policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))
    m.save(str(C.CHECKPOINT_DIR / "single_a"))
    m.save(str(C.CHECKPOINT_DIR / "single_b"))
    return "single_a"


def write_eval(name, score):
    (C.MODEL_CONFIG_DIR / f"{name}.eval.json").write_text(json.dumps(dict(score=score, episodes=10, levels=[dict(level=0, success=int(10 * score), n=10)])))


def test_the_default_head_start_is_the_best_tested_single_lander_network(sandbox, single):
    write_eval("single_a", 0.4)
    write_eval("single_b", 0.9)
    assert startup._best_single() == "single_b"
    assert startup.status()["suggested_model"] == "single_b"


def test_copy_head_start_for_a_pair_is_instant_and_training_follows(sandbox, single):
    _, calls = sandbox
    code, res = startup.begin(dict(mode="pair", source="copy", from_model=single, then_train=True, name="t1",
                                   train=dict(total_steps=500_000, n_envs=2, learning_rate=1e-4, start_level=0), dual=dict(spacing=800)))
    assert code == 200 and res["eta_min"] == 0 and res["pipeline"] is None
    assert (C.CHECKPOINT_DIR / "t1_final_A.zip").exists() and (C.CHECKPOINT_DIR / "t1_final_B.zip").exists()
    p = calls["start"][0]
    assert p["start_profile"] == "pair" and p["init_from"] == "t1_final" and p["n_envs"] == 2 and p["total_steps"] == 500_000 and p["learning_rate"] == 1e-4
    assert p["ent_coef"] == 0 and p["staged"] == "on" and p["dual_spacing"] == 800.0
    calls["running"] = True                                                                                         # the trainer is now busy
    st = startup.status()
    assert [s["key"] for s in st["steps"]] == ["head", "train"] and st["steps"][0]["state"] == "done" and st["steps"][1]["state"] == "running" and st["mode"] == "pair"


def test_copy_for_one_network_dual_and_single_and_blank(sandbox, single):
    _, calls = sandbox
    startup.begin(dict(mode="dual", source="copy", from_model=single, then_train=False, name="d1"))
    assert W.checkpoint_info("d1_final")["kind"] == "dual" and not calls["start"]
    startup.begin(dict(mode="single", source="copy", from_model=single, then_train=True, name="s1"))
    assert calls["start"][-1]["start_profile"] == "spacex" and calls["start"][-1]["init_from"] == single
    startup.begin(dict(mode="pair", source="blank", then_train=True, name="b1"))
    assert "init_from" not in calls["start"][-1] and startup.status()["steps"][0]["state"] == "skipped"


def test_bad_requests_are_refused_with_a_reason(sandbox, single):
    _, calls = sandbox
    with pytest.raises(ValueError, match="mode"):
        startup.begin(dict(mode="triple", source="copy", from_model=single))
    with pytest.raises(ValueError, match="no model"):
        startup.begin(dict(mode="pair", source="copy", from_model="ghost"))
    with pytest.raises(ValueError, match="between"):
        startup.begin(dict(mode="pair", source="copy", from_model=single, train=dict(learning_rate=0.5)))          # unsafe learning rate
    with pytest.raises(ValueError, match="between"):
        startup.begin(dict(mode="pair", source="copy", from_model=single, quality=9))
    startup.begin(dict(mode="pair", source="copy", from_model=single, then_train=False, name="k1"))
    with pytest.raises(ValueError, match="not a single-lander"):
        startup.begin(dict(mode="pair", source="copy", from_model="k1_final_A"))                                      # one of a pair is not a single-lander brain
    calls["running"] = True
    code, res = startup.begin(dict(mode="pair", source="copy", from_model=single, then_train=True, name="k2"))
    assert code == 409 and "already running" in res["errors"][0]


class FakeProc:
    def __init__(self):
        self.code = None
        self.pid = 1

    def poll(self):
        return self.code


def test_the_expert_pipeline_runs_and_training_starts_by_itself_when_it_is_done(sandbox, single, monkeypatch):
    tmp, calls = sandbox
    proc = FakeProc()
    seen = {}

    def fake_popen(argv, out, env_extra=None):
        seen["argv"], seen["env"] = argv, env_extra
        return proc
    monkeypatch.setattr(jobs, "_popen", fake_popen)
    code, res = startup.begin(dict(mode="pair", source="expert", from_model=single, quality=1, then_train=True, name="e1",
                                   train=dict(total_steps=300_000, n_envs=3, learning_rate=5e-5, start_level=1), dual=dict(spacing=700)))
    assert code == 200 and res["pipeline"] == "e1" and res["eta_min"] == 18                                        # quick, two landers: 12 x 1.5
    cfg = json.loads((tmp / "logs" / "pipeline" / "e1" / "config.json").read_text())
    assert cfg["pair"] is True and cfg["from_single"] == single and cfg["steps"] == ["expert", "dagger", "eval"]       # starting from a brain: no cloning step
    assert cfg["expert"]["episodes"] == 150 and cfg["dagger"]["rounds"] == 2
    assert json.loads(seen["env"]["ROCKETEER_OVERRIDES"])["config"]["DUAL_PAD_SPACING_M"] == 700.0                 # the pad spacing reaches the pipeline process
    time.sleep(0.3)
    assert not calls["start"]                                                                                      # still waiting for the head start
    st = startup.status()
    assert st["state"] in ("running", "waiting") and st["steps"][1]["state"] == "pending"
    (tmp / "logs" / "pipeline" / "e1" / "status.json").write_text(json.dumps(dict(name="e1", state="done", final="e1_final", steps=[], rounds=[], final_eval=dict(score=0.8))))
    proc.code = 0
    for _ in range(40):
        if calls["start"]:
            break
        time.sleep(0.25)
    assert calls["start"] and calls["start"][0]["init_from"] == "e1_final" and calls["start"][0]["start_profile"] == "pair" and calls["start"][0]["start_level"] == 1
    calls["running"] = True
    assert startup.status()["steps"][1]["state"] == "running"


def test_a_failed_head_start_does_not_start_training_and_says_so(sandbox, single, monkeypatch):
    tmp, calls = sandbox
    proc = FakeProc()
    monkeypatch.setattr(jobs, "_popen", lambda argv, out, env_extra=None: proc)
    startup.begin(dict(mode="single", source="expert", from_model="", quality=1, then_train=True, name="f1"))
    cfg = json.loads((tmp / "logs" / "pipeline" / "f1" / "config.json").read_text())
    assert cfg["steps"] == ["expert", "clone", "dagger", "eval"] and cfg["init"] == ""                              # from scratch: the cloning step is needed
    (tmp / "logs" / "pipeline" / "f1" / "status.json").write_text(json.dumps(dict(name="f1", state="failed", final=None, steps=[], rounds=[])))
    proc.code = 1
    for _ in range(40):
        if startup.status().get("error"):
            break
        time.sleep(0.25)
    assert not calls["start"] and "did not finish" in startup.status()["error"]


def test_stop_stops_the_pipeline_and_the_training(sandbox, single):
    _, calls = sandbox
    startup.begin(dict(mode="pair", source="copy", from_model=single, then_train=False, name="z1"))
    out = startup.stop()
    assert calls["stop"] == 1 and "pipeline" in out


def test_an_existing_pair_can_be_the_head_start_of_a_pair_run_as_it_is(sandbox, single):
    from src.training.pair import make_pair
    _, calls = sandbox
    make_pair(single, "mypair")
    code, res = startup.begin(dict(mode="pair", source="copy", from_model="mypair", then_train=True, name="t9", train=dict(total_steps=500_000, n_envs=2, learning_rate=1e-4, start_level=0)))
    assert code == 200
    assert calls["start"][0]["start_profile"] == "pair" and calls["start"][0]["init_from"] == "mypair"
    assert not (C.CHECKPOINT_DIR / "t9_final_A.zip").exists()                      # nothing was copied: the pair is used as it is
    assert "as it is" in startup.status()["steps"][0]["detail"]


def test_a_pair_cannot_start_a_single_dual_or_expert_run(sandbox, single):
    from src.training.pair import make_pair
    make_pair(single, "mypair")
    for mode, source in (("single", "copy"), ("dual", "copy"), ("pair", "expert")):
        with pytest.raises(ValueError, match="pair of brains"):
            startup.begin(dict(mode=mode, source=source, from_model="mypair", then_train=False))
