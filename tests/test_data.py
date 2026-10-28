import csv
import gzip
import io
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from src.training import data as D
from src.training import finetune as FT
from src.training import weights_io as W
from src.training.env import make_vec_env
from src.utils import config as C


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "ROOT_DIR", tmp_path / "traj")
    monkeypatch.setattr(C, "CHECKPOINT_DIR", tmp_path / "ckpt")
    monkeypatch.setattr(C, "MODEL_CONFIG_DIR", tmp_path / "cfg")
    (tmp_path / "ckpt").mkdir()
    return tmp_path


def fake_episode(logger, env, T, rng, ret_scale=1.0, merit=10.0, soft=False, f=None):
    """T steps with action = f(obs) so the data has structure to learn from."""
    f = f or (lambda o: np.array([np.clip(o[0], -1, 1), np.clip(-o[1], -1, 1), 0.0, 0.0, 0.0], np.float32))
    for t in range(T):
        o = rng.normal(size=D.N_OBS).astype(np.float32)
        logger.step(env, o, f(o), ret_scale * (1.0 if t < T - 1 else 5.0))
    logger.end_episode(env, dict(level=0, rules_stage=0, end="rest" if soft else "crash", merit=merit, soft=soft, success=soft), 1000 * T)


def make_logger(roots, **kw):
    return D.TrajectoryLogger(C.LOG_DIR / "unit_run", "unit_run", **kw)


def test_logger_keeps_samples_and_the_tails_and_writes_a_manifest(roots):
    rng = np.random.default_rng(0)
    lg = make_logger(roots, every=10, tail=0.1, flush_episodes=1000)
    for i in range(400):
        fake_episode(lg, i % 3, 5, rng, ret_scale=float(rng.uniform(-5, 5)))
    lg.flush()
    rows = list(csv.DictReader(open(lg.dir / "episodes_kept.csv")))
    why = [r["kept_for"] for r in rows]
    assert 40 <= sum("sample" in w for w in why) <= 41                                   # every 10th of 400
    assert any("best" in w for w in why) and any("worst" in w for w in why)
    assert len(rows) < 400 * 0.4                                                          # sampled, not everything
    man = json.loads((lg.dir / "manifest.json").read_text())
    assert man["columns"] == D.STEP_COLS and len(man["reward_fingerprint"]) == 12 and man["seen"] == 400
    with gzip.open(next(lg.dir.glob("traj_*.csv.gz")), "rt") as f:
        head = next(csv.reader(f))
    assert head == D.STEP_COLS
    lg2 = make_logger(roots, every=1)                                                     # a resumed run keeps numbering instead of overwriting
    fake_episode(lg2, 0, 4, rng)
    lg2.flush()
    assert len(list(lg.dir.glob("traj_*.csv.gz"))) == 2 and int(list(csv.DictReader(open(lg.dir / "episodes_kept.csv")))[-1]["ep"]) == len(rows) + 1


def test_budget_deletes_the_oldest_chunks_first(roots):
    rng = np.random.default_rng(1)
    lg = make_logger(roots, every=1, flush_episodes=5, budget_gb=1e-9)                    # a budget of ~1 byte: only the newest chunk survives
    for i in range(30):
        fake_episode(lg, 0, 20, rng)
    lg.flush()
    left = sorted(lg.dir.glob("traj_*.csv.gz"))
    assert len(left) == 1 and left[0].name == f"traj_{lg.chunk:05d}.csv.gz"


def test_dataset_filters_and_returns(roots):
    rng = np.random.default_rng(2)
    lg = make_logger(roots, every=1, flush_episodes=1000)
    for i in range(12):
        fake_episode(lg, 0, 6, rng, merit=60.0 if i % 4 == 0 else 5.0, soft=(i % 4 == 0))
    lg.flush()
    allds = D.load_dataset("unit_run", gamma=0.5)
    assert allds["episodes"] == 12 and allds["obs"].shape == (72, D.N_OBS) and allds["act"].shape == (72, D.N_ACT)
    assert D.load_dataset("unit_run", min_merit=50)["episodes"] == 3
    assert D.load_dataset("unit_run", soft_only=True)["episodes"] == 3
    assert D.load_dataset("unit_run", min_merit=100)["episodes"] == 0
    # discounted return of an episode with rewards [1,1,1,1,1,5], gamma 0.5: last step 5, one before 1 + 2.5 ...
    r = D.discounted_returns(np.array([1, 1, 1, 1, 1, 5], np.float32), 0.5)
    assert r[-1] == 5 and r[-2] == pytest.approx(3.5) and r[0] == pytest.approx(1 + 0.5 * r[1])


def test_flat_csv_round_trips_and_keeps_the_filters_working(roots):
    rng = np.random.default_rng(3)
    lg = make_logger(roots, every=1, flush_episodes=1000)
    for i in range(10):
        fake_episode(lg, 0, 7, rng, merit=70.0 if i < 3 else 2.0, soft=i < 3)
    lg.flush()
    out = roots / "export.csv"
    counts = D.export_flat_csv("unit_run", out)
    assert counts == dict(episodes=10, steps=70)
    head = next(csv.reader(open(out)))
    assert head[:len(D.STEP_COLS)] == D.STEP_COLS and "merit" in head
    a, b = D.load_dataset("unit_run"), D.load_dataset(out)
    assert b["episodes"] == 10 and np.allclose(a["obs"], b["obs"], atol=1e-4) and np.allclose(a["act"], b["act"], atol=1e-4)
    assert D.load_dataset(out, soft_only=True)["episodes"] == 3                           # the outcome travels with the file
    assert D.load_dataset(out, min_merit=50)["episodes"] == 3
    buf = io.StringIO()
    assert D.export_flat_csv("unit_run", buf, soft_only=True)["episodes"] == 3


@pytest.fixture
def checkpoint(roots):
    vec = make_vec_env(1, 0, level=0, reward="custom")
    model = PPO("MlpPolicy", vec, device="cpu", verbose=0, seed=0)
    model.save(str(C.CHECKPOINT_DIR / "unit_model"))
    vec.close()
    return "unit_model"


def policy_tensors(name):
    m = PPO.load(str(C.CHECKPOINT_DIR / f"{name}.zip"), device="cpu")
    return {k: v.detach().numpy().copy() for k, v in m.policy.state_dict().items()}


def test_weights_csv_round_trip_is_exact_and_backs_up(roots, checkpoint):
    out = roots / "w.csv"
    info = W.export_weights_csv(checkpoint, out)
    assert info["values"] > 9000
    before = policy_tensors(checkpoint)
    rows = list(csv.reader(open(out)))
    assert rows[0] == ["tensor", "shape", "index", "value"] and len(rows) - 1 == info["values"]
    # edit one weight in the CSV (a "hand edit") and import into a NEW checkpoint: exact copy otherwise
    idx = next(i for i, r in enumerate(rows) if r[0] == "action_net.weight" and r[2] == "3")
    rows[idx][3] = "0.5"
    edited = roots / "w_edited.csv"
    csv.writer(open(edited, "w", newline="")).writerows(rows)
    res = W.import_weights_csv(edited, checkpoint, out_name="unit_model_edit")
    assert res["backup"] is None
    after = policy_tensors("unit_model_edit")
    for k in before:
        flat_a, flat_b = after[k].ravel(), before[k].ravel()
        if k == "action_net.weight":
            assert flat_a[3] == pytest.approx(0.5) and np.array_equal(np.delete(flat_a, 3), np.delete(flat_b, 3))
        else:
            assert np.array_equal(flat_a, flat_b), k
    assert np.array_equal(policy_tensors(checkpoint)["action_net.weight"], before["action_net.weight"])   # the source was not touched
    # overwrite mode: backup first
    res = W.import_weights_csv(edited, checkpoint)
    assert res["backup"] and (C.CHECKPOINT_DIR / res["backup"]).exists()
    assert policy_tensors(checkpoint)["action_net.weight"].ravel()[3] == pytest.approx(0.5)
    assert np.array_equal(policy_tensors(res["backup"][:-4])["action_net.weight"], before["action_net.weight"])


def test_weights_import_rejects_files_that_do_not_fit(roots, checkpoint):
    bad = roots / "bad.csv"
    bad.write_text("tensor,shape,index,value\naction_net.weight,2x64,0,1.0\n")
    with pytest.raises(ValueError):
        W.import_weights_csv(bad, checkpoint, out_name="x")
    wrong = roots / "wrong.csv"
    wrong.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        W.read_weights_csv(wrong)
    full = roots / "full.csv"
    W.export_weights_csv(checkpoint, full)
    lines = full.read_text().splitlines()
    full.write_text("\n".join(lines[:-3]) + "\n")                                        # truncated file: missing indices
    with pytest.raises(ValueError):
        W.import_weights_csv(full, checkpoint, out_name="x")
    assert not (C.CHECKPOINT_DIR / "x.zip").exists()


def test_cloning_learns_the_logged_behaviour_and_leaves_the_source_alone(roots, checkpoint):
    rng = np.random.default_rng(4)
    lg = make_logger(roots, every=1, flush_episodes=1000)
    for i in range(60):
        fake_episode(lg, 0, 25, rng, merit=50.0 if i % 2 else 1.0)
    lg.flush()
    src_before = policy_tensors(checkpoint)
    rep = FT.finetune(checkpoint, ["unit_run"], "unit_clone", epochs=25, lr=3e-3, min_merit=10, log=lambda s: None)
    assert rep["episodes"] == 30 and rep["steps"] == 750
    assert rep["after"]["bc"] < 0.25 * rep["before"]["bc"]                               # the policy now imitates the data
    assert rep["after"]["value"] < rep["before"]["value"]
    for k, v in policy_tensors(checkpoint).items():
        assert np.array_equal(v, src_before[k])                                          # source checkpoint unchanged
    clone = PPO.load(str(C.CHECKPOINT_DIR / "unit_clone.zip"), device="cpu")
    o = rng.normal(size=(5, D.N_OBS)).astype(np.float32)
    mean = clone.policy.get_distribution(torch.as_tensor(o)).distribution.mean.detach().numpy()
    assert np.abs(mean[:, 0] - np.clip(o[:, 0], -1, 1)).mean() < 0.25
    assert (C.MODEL_CONFIG_DIR / "unit_clone.json").exists()
    with pytest.raises(ValueError):
        FT.finetune(checkpoint, ["unit_run"], "unit_none", min_merit=1000, log=lambda s: None)


def test_the_trainer_writes_a_trajectory_log_and_can_start_from_a_checkpoint():
    tag = "_datatest"
    name = f"custom_s0{tag}"
    tdir = D.ROOT_DIR / name
    overrides = {"config": {"FUEL_BUDGET": 100000.0, "DATA_EVERY": 1}}
    env = dict(os.environ, ROCKETEER_OVERRIDES=json.dumps(overrides))
    try:
        out = subprocess.run([sys.executable, "src/training/train.py", "--reward", "custom", "--seed", "0", "--total-steps", "6000", "--n-envs", "2",
                              "--tag", tag], cwd=C.ROOT, capture_output=True, text=True, timeout=300, env=env)
        assert out.returncode == 0, out.stderr[-600:]
        assert any(tdir.glob("traj_*.csv.gz")) and (tdir / "manifest.json").exists()
        ds = D.load_dataset(name)
        assert ds["episodes"] >= 3 and ds["obs"].shape[1] == D.N_OBS and ds["act"].shape[1] == D.N_ACT
        assert np.abs(ds["act"]).max() <= 1.0 + 1e-6
        x0 = ds["obs"][0]
        assert x0[6] == pytest.approx(1.0) and x0[7] == 0.0 and x0[11] == 0.0                # first step: full tank, thrust zero, engine off
        out2 = subprocess.run([sys.executable, "src/training/train.py", "--reward", "custom", "--seed", "1", "--total-steps", "2000", "--n-envs", "1",
                               "--tag", tag + "2", "--init-from", name], cwd=C.ROOT, capture_output=True, text=True, timeout=300, env=env)
        assert out2.returncode == 0 and f"weights loaded from {name}.zip" in out2.stdout
    finally:
        for run in (name, f"custom_s1{tag}2"):
            shutil.rmtree(C.LOG_DIR / run, ignore_errors=True)
            shutil.rmtree(D.ROOT_DIR / run, ignore_errors=True)
            (C.LOG_DIR / f"{run}.out").unlink(missing_ok=True)
            for p in list(C.MODEL_CONFIG_DIR.glob(f"{run}*")) + list(C.CHECKPOINT_DIR.glob(f"{run}*")):
                p.unlink(missing_ok=True)


def test_server_data_helpers_validate_their_input(roots, monkeypatch, checkpoint):
    from src.app import server
    monkeypatch.setattr(server, "IMPORTS_DIR", roots / "imports")
    up = server.save_upload(dict(name="../../evil name.csv", text="tensor,shape,index,value\nx,1,0,0.5\n"))
    assert up["kind"] == "weights" and (roots / "imports" / up["name"]).is_file() and ".." not in up["name"]
    assert server.save_upload(dict(name="t.csv", text="ep,t,o0,o1\n"))["kind"] == "trajectories"
    with pytest.raises(ValueError):
        server.save_upload(dict(name="x.csv", text="a,b\n1,2\n"))
    for bad in ("../x", "a/b", "", "x" * 80):
        with pytest.raises(FileNotFoundError):
            server.data_source(bad)
    assert server._flt({"min_merit": ["25"], "soft_only": ["1"], "kept": ["best+sample"], "max_return": [""]}) == dict(min_merit=25.0, soft_only=True, kept_for=["best", "sample"])
    # the clone job refuses bad input before starting anything
    code, res = server.start_finetune(dict(checkpoint="nope", source="x", out="y"))
    assert code == 400
    code, res = server.start_finetune(dict(checkpoint=checkpoint, source="x", out=checkpoint))
    assert code == 409 and "never overwrites" in res["errors"][0]
    code, res = server.start_finetune(dict(checkpoint=checkpoint, source="x", out="../bad"))
    assert code == 400
    code, res = server.import_weights(dict(file="missing.csv", checkpoint=checkpoint))
    assert code == 404
