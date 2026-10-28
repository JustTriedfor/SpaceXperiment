"""The imitation pipeline, the checkpoint tools and the dimension-aware data readers (single lander and Dual land)."""
import json

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.app import jobs
from src.sim import pipeline as PL
from src.sim.env import SpaceXEnv
from src.training import data as D
from src.training import weights_io as W
from src.utils import config as C


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Every folder the pipeline touches, redirected into tmp_path."""
    for name, sub in (("LOG_DIR", "logs"), ("CHECKPOINT_DIR", "ckpt"), ("MODEL_CONFIG_DIR", "cfg"), ("DATA_DIR", "data")):
        monkeypatch.setattr(C, name, tmp_path / sub)
        (tmp_path / sub).mkdir()
    monkeypatch.setattr(D, "ROOT_DIR", tmp_path / "data" / "trajectories")
    monkeypatch.setattr(jobs, "PIPE_DIR", tmp_path / "logs" / "pipeline")
    monkeypatch.setattr(jobs, "TRASH_DIR", tmp_path / "trash")
    W._INFO_CACHE.clear()
    return tmp_path


@pytest.fixture
def single_ckpt(sandbox):
    m = PPO("MlpPolicy", DummyVecEnv([lambda: SpaceXEnv()]), seed=0, device="cpu", policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))
    m.save(str(C.CHECKPOINT_DIR / "single_a"))
    return "single_a"


def test_logged_data_carries_its_own_dimensions_and_mixing_is_refused(sandbox):
    rng = np.random.default_rng(0)
    for run, (n_obs, n_act) in (("one", (18, 5)), ("two", (40, 10))):
        lg = D.TrajectoryLogger(C.LOG_DIR / run, run, every=1)
        for _ in range(2):
            for t in range(5):
                lg.step(0, rng.normal(size=n_obs), rng.normal(size=n_act), 0.1)
            lg.end_episode(0, dict(level=0, rules_stage=0, end="rest", merit=5.0, soft=True, success=True), 100)
        lg.flush()
    assert D.dataset_info("one") == dict(n_obs=18, n_act=5, kind="single") and D.dataset_info("two") == dict(n_obs=40, n_act=10, kind="dual")
    a, b = D.load_dataset("one"), D.load_dataset("two")
    assert a["obs"].shape == (10, 18) and a["act"].shape == (10, 5) and b["obs"].shape == (10, 40) and b["act"].shape == (10, 10)
    with pytest.raises(ValueError, match="mixes"):
        D.load_dataset(["one", "two"])
    kinds = {r["run"]: r["kind"] for r in D.run_summaries()}
    assert kinds == {"one": "single", "two": "dual"}
    out = sandbox / "flat.csv"
    D.export_flat_csv("two", out)
    head = out.read_text().splitlines()[0].split(",")
    assert "o39" in head and "a9" in head and "o40" not in head
    assert D.dataset_info(str(out))["kind"] == "dual" and D.load_dataset(str(out))["obs"].shape == (10, 40)         # a flat file round-trips too


def test_checkpoint_info_dual_copy_quiet_copy_and_trash(sandbox, single_ckpt):
    info = W.checkpoint_info(single_ckpt)
    assert info["kind"] == "single" and (info["n_obs"], info["n_act"]) == (18, 5) and info["arch"] == [256, 256] and info["std"] == pytest.approx(1.0, abs=0.01)
    r = jobs.ckpt_make_dual(dict(**{"from": single_ckpt}, out="dual_a"))
    d = W.checkpoint_info("dual_a")
    assert r["checkpoint"] == "dual_a.zip" and d["kind"] == "dual" and (d["n_obs"], d["n_act"]) == (40, 10) and d["arch"] == [512, 512]
    with pytest.raises(ValueError, match="already exists"):
        jobs.ckpt_make_dual(dict(**{"from": single_ckpt}, out="dual_a"))
    with pytest.raises(ValueError, match="not a single-lander"):
        jobs.ckpt_make_dual(dict(**{"from": "dual_a"}, out="dual_b"))
    jobs.ckpt_quiet(dict(**{"from": single_ckpt}, out="quiet_a", std=0.08))
    assert W.checkpoint_info("quiet_a")["std"] == pytest.approx(0.08, abs=0.002) and W.checkpoint_info(single_ckpt)["std"] == pytest.approx(1.0, abs=0.01)    # the source is untouched
    with pytest.raises(ValueError):
        jobs.ckpt_quiet(dict(**{"from": single_ckpt}, out="quiet_b", std=3.0))
    names = {c["name"]: c for c in jobs.checkpoint_details("quiet")}
    assert names["quiet_a"]["busy"] and not names["dual_a"]["busy"] and names["dual_a"]["kind"] == "dual"
    with pytest.raises(ValueError, match="training right now"):
        jobs.ckpt_delete(dict(name="quiet_a"), training_run="quiet")
    jobs.ckpt_delete(dict(name="dual_a"))
    assert not (C.CHECKPOINT_DIR / "dual_a.zip").exists() and list((sandbox / "trash").rglob("dual_a.zip"))            # moved, not deleted


def test_cloning_refuses_data_of_the_other_kind(sandbox, single_ckpt):
    from src.training.finetune import finetune
    rng = np.random.default_rng(1)
    lg = D.TrajectoryLogger(C.LOG_DIR / "dualdata", "dualdata", every=1)
    for t in range(6):
        lg.step(0, rng.normal(size=40), rng.normal(size=10), 0.1)
    lg.end_episode(0, dict(level=0, rules_stage=0, end="rest", merit=5.0, soft=True, success=True), 100)
    lg.flush()
    with pytest.raises(ValueError, match="Dual land"):
        finetune(single_ckpt, ["dualdata"], "x_clone", epochs=1, log=lambda s: None)


def test_pipeline_validation_and_name_clashes(sandbox, single_ckpt, monkeypatch):
    monkeypatch.setattr(jobs, "_popen", lambda argv, out, env_extra=None: type("P", (), {"poll": lambda self: None, "pid": 1})())
    with pytest.raises(ValueError, match="name"):
        jobs.pipeline_start(dict(name="x"))
    with pytest.raises(ValueError, match="levels"):
        jobs.pipeline_start(dict(name="okname", expert=dict(levels="0,9")))
    with pytest.raises(ValueError, match="episodes"):
        jobs.pipeline_start(dict(name="okname", expert=dict(episodes=1)))
    with pytest.raises(ValueError, match="single-lander checkpoint"):
        jobs.pipeline_start(dict(name="okname", dual=False, from_single=single_ckpt))               # only dual pipelines start from a single one
    jobs.ckpt_make_dual(dict(**{"from": single_ckpt}, out="dual_a"))
    with pytest.raises(ValueError, match="this pipeline is single"):
        jobs.pipeline_start(dict(name="okname", dual=False, init="dual_a"))
    with pytest.raises(ValueError, match="no demonstrations"):
        jobs.pipeline_start(dict(name="okname", steps=["clone"]))
    code, res = jobs.pipeline_start(dict(name="okname", dual=True, from_single=single_ckpt))
    assert code == 200 and res["dual"] is True and res["steps"] == ["expert", "clone", "dagger", "eval"]
    cfg = json.loads((sandbox / "logs" / "pipeline" / "okname" / "config.json").read_text())
    assert cfg["from_single"] == single_ckpt and cfg["dual"] is True and cfg["dagger"]["rounds"] == 6
    jobs.PIPE.update(proc=None, name=None)
    (C.CHECKPOINT_DIR / "okname_final.zip").write_bytes(b"x")                                       # a leftover from an earlier run with this name
    code, res = jobs.pipeline_start(dict(name="okname", dual=True))
    assert code == 409 and "already has outputs" in res["errors"][0]
    jobs.PIPE.update(proc=None, name=None)


def test_a_whole_single_lander_pipeline_runs_and_writes_its_status(sandbox):
    cfg = dict(name="tiny", dual=False, expert=dict(episodes=3, noise=0.1, levels="0"), clone=dict(epochs=1, lr=1e-3, batch=256),
               dagger=dict(rounds=1, episodes=2, epochs=1, levels="0"), eval=dict(episodes=2, levels="0"))
    st = PL.run(cfg)
    assert st["state"] == "done" and st["final"] == "tiny_final" and st["best"] == "tiny_dagger_1" and len(st["rounds"]) == 1
    assert all(s["state"] == "done" for s in st["steps"])
    for f in ("tiny_init", "tiny_clone", "tiny_dagger_1", "tiny_final"):
        assert (C.CHECKPOINT_DIR / f"{f}.zip").exists()
    assert D.dataset_info("tiny_expert")["kind"] == "single" and D.dataset_info("tiny_dagger_1")["kind"] == "single"
    assert json.loads((C.MODEL_CONFIG_DIR / "tiny_final.eval.json").read_text())["kind"] == "single"
    on_disk = json.loads((C.LOG_DIR / "pipeline" / "tiny" / "status.json").read_text())
    assert on_disk["state"] == "done" and on_disk["final_eval"]["levels"][0]["n"] == 2
    assert PL.outputs_of("tiny")["checkpoints"] and PL.outputs_of("tiny")["data"]
    PL.clear_outputs("tiny")
    assert not any(PL.outputs_of("tiny").values())


def test_the_dual_pipeline_collects_pairs_and_starts_from_two_copies(sandbox, single_ckpt):
    cfg = dict(name="twin", dual=True, from_single=single_ckpt, steps=["expert", "clone", "eval"], expert=dict(episodes=2, noise=0.0, levels="0"),
               clone=dict(epochs=1, lr=1e-4, batch=256), eval=dict(episodes=2, levels="0"))
    st = PL.run(cfg)
    assert st["state"] == "done" and st["final"] == "twin_final"
    assert W.checkpoint_info("twin_final")["kind"] == "dual" and W.checkpoint_info("twin_init")["kind"] == "dual"
    assert D.dataset_info("twin_expert") == dict(n_obs=40, n_act=10, kind="dual")
    assert st["final_eval"]["kind"] == "dual" and "gap_median" in st["final_eval"]["levels"][0]


def test_the_pair_pipeline_gives_each_network_its_own_data_and_ends_with_a_pair(sandbox, single_ckpt):
    from src.training.pair import resolve_pair
    cfg = dict(name="duo", pair=True, from_single=single_ckpt, expert=dict(episodes=3, noise=0.0, levels="0"), clone=dict(epochs=1, lr=1e-4, batch=256),
               dagger=dict(rounds=1, episodes=2, epochs=1, lr=1e-4, levels="0"), eval=dict(episodes=2, levels="0"))
    st = PL.run(cfg)
    assert st["state"] == "done" and st["final"] == "duo_final" and st["best"] == "duo_dagger_1"
    assert all(s["state"] == "done" for s in st["steps"])
    for base in ("duo_init", "duo_clone", "duo_dagger_1", "duo_final"):
        assert resolve_pair(base) is not None, base                                             # every stage is a pair of networks
        assert W.checkpoint_info(f"{base}_A")["kind"] == "agent" and W.checkpoint_info(f"{base}_B")["kind"] == "agent"
    for d in ("duo_expert_A", "duo_expert_B", "duo_dagger_1_A", "duo_dagger_1_B"):
        assert D.dataset_info(d) == dict(n_obs=40, n_act=5, kind="agent"), d                     # each network's own lander view and 5 actions
    a, b = D.load_dataset("duo_expert_A"), D.load_dataset("duo_expert_B")
    assert a["obs"].shape == b["obs"].shape and a["obs"].shape[1] == 40 and a["act"].shape[1] == 5
    assert np.allclose(a["obs"][:, :18], b["obs"][:, 18:36]) and np.allclose(a["obs"][:, 18:36], b["obs"][:, :18])      # B sees the same world from its own side: own and partner swapped
    assert st["final_eval"]["kind"] == "pair" and "gap_median" in st["final_eval"]["levels"][0]
    assert any(PL.outputs_of("duo").values())
    PL.clear_outputs("duo")
    assert not any(PL.outputs_of("duo").values())


def test_pair_pipeline_validation(sandbox, single_ckpt, monkeypatch):
    from src.training.pair import make_pair
    monkeypatch.setattr(jobs, "_popen", lambda argv, out, env_extra=None: type("P", (), {"poll": lambda self: None, "pid": 1})())
    make_pair(single_ckpt, "prs")
    with pytest.raises(ValueError, match="pair"):
        jobs.pipeline_start(dict(name="okpair", pair=True, init=single_ckpt))                    # a single-lander checkpoint is not a pair
    with pytest.raises(ValueError, match="single"):
        jobs.pipeline_start(dict(name="okpair", pair=False, dual=False, init="prs"))              # a pair cannot start a single-lander pipeline
    code, res = jobs.pipeline_start(dict(name="okpair", pair=True, init="prs"))
    assert code == 200 and res["pair"] is True and res["dual"] is False
    cfg = json.loads((sandbox / "logs" / "pipeline" / "okpair" / "config.json").read_text())
    assert cfg["pair"] is True and cfg["init"] == "prs"
    jobs.PIPE.update(proc=None, name=None)
    code, res = jobs.pipeline_start(dict(name="okpair2", pair=True, from_single=single_ckpt))
    assert code == 200
    jobs.PIPE.update(proc=None, name=None)
