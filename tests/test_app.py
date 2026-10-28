import json
import os
import subprocess
import sys

from src.app.schema import build_run, public_schema


def test_defaults_build_clean_run():
    r = build_run({})
    assert "errors" not in r
    assert r["name"] == "custom_s0_app"
    assert r["overrides"] == {"config": {"START_PROFILE": "spacex", "RULES_STAGE_START": 0}, "reward": {}}   # app defaults
    assert "--stop-at-level" not in r["argv"]


def test_changed_values_become_overrides():
    r = build_run({"fuel_budget": 4000, "ent_coef": 0.02, "r_timeout": -300, "stop_at_level": 1, "reward": "custom"})
    assert r["overrides"]["config"]["FUEL_BUDGET"] == 4000
    assert r["overrides"]["config"]["PPO_KWARGS"] == {"ent_coef": 0.02}
    assert r["overrides"]["reward"]["TIMEOUT_PENALTY"] == -300
    assert r["argv"][r["argv"].index("--stop-at-level") + 1] == "1"


def test_only_the_custom_reward_exists():
    assert "errors" in build_run({"reward": "default"})                       # SpaceXperiment has one reward type
    assert build_run({"reward": "custom", "r_timeout": -300})["overrides"]["reward"] == {"TIMEOUT_PENALTY": -300.0}


def test_validation():
    assert "errors" in build_run({"batch_size": 100})                 # does not divide 1024 x 8
    assert "errors" in build_run({"tag": "../evil"})
    assert "errors" in build_run({"n_envs": 99})
    assert "errors" in build_run({"reward": "nope"})
    assert "errors" in build_run({"start_level": 2, "stop_at_level": 1})
    assert len(public_schema()) >= 25


def test_overrides_reach_a_fresh_process():
    ov = {"config": {"FUEL_BUDGET": 12.5, "PPO_KWARGS": {"ent_coef": 0.5}}, "reward": {"TIMEOUT_PENALTY": -7.0}}
    code = "from src.training import reward as R; from src.utils import config as C; " \
           "print(C.FUEL_BUDGET, C.PPO_KWARGS['ent_coef'], C.PPO_KWARGS['n_steps'], R.TIMEOUT_PENALTY)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=dict(os.environ, ROCKETEER_OVERRIDES=json.dumps(ov))).stdout.split()
    assert out == ["12.5", "0.5", "1024", "-7.0"]


def test_lr_growth_is_exponential():
    from src.utils.config import agent_lrs
    lrs = agent_lrs(1e-4, 2.0, 5)
    assert lrs == [1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3]
    assert agent_lrs(3e-4, 1.0, 3) == [3e-4] * 3


def test_sweep_mode_uses_sweep_script():
    r = build_run({"sweep": "on", "n_envs": 4, "lr_growth": 1.5, "tag": "_x"})
    assert r["argv"][1] == "src/training/sweep.py" and r["name"] == "custom_s0_x_sweep"
    assert r["argv"][r["argv"].index("--lr-growth") + 1] == "1.5"
    assert build_run({})["argv"][1] == "src/training/train.py"


def test_sweep_rejects_runaway_learning_rates():
    r = build_run({"sweep": "on", "n_envs": 16, "lr_growth": 2.0})
    assert "errors" in r and "learning rates" in r["errors"][0]


def test_sweep_batch_check_uses_one_env_per_agent():
    # 1024 x 8 envs is divisible by 2048, but each sweep agent only collects 1024 per rollout
    assert "errors" not in build_run({"batch_size": 2048})
    assert "errors" in build_run({"batch_size": 2048, "sweep": "on"})


def test_sweep_copies_group_agents():
    r = build_run({"sweep": "on", "n_envs": 32, "sweep_copies": 4, "total_steps": 250000})
    assert r["argv"][r["argv"].index("--copies") + 1] == "4" and r["argv"][r["argv"].index("--n-envs") + 1] == "32"
    assert "errors" in build_run({"sweep": "on", "n_envs": 30, "sweep_copies": 4})


def test_app_defaults_to_spacex_profile_override():
    assert build_run({})["overrides"]["config"]["START_PROFILE"] == "spacex"
    assert "errors" in build_run({"start_profile": "classic"})                  # only the re-entry start exists now


def test_speed_ramp_override():
    r = build_run({"ramp_start": 0.25})
    assert r["overrides"]["config"]["SPACEX_LEVEL_SCALE"] == [0.25, 0.5, 0.75, 1.0]
    assert "SPACEX_LEVEL_SCALE" not in build_run({"ramp_start": 1.0})["overrides"]["config"]
    assert "errors" in build_run({"ramp_start": 0.0})


def test_device_option_defaults_to_cpu_and_overrides_when_changed():
    assert "DEVICE" not in build_run({})["overrides"]["config"]
    assert build_run({"device": "cuda"})["overrides"]["config"]["DEVICE"] == "cuda"
    assert "errors" in build_run({"device": "tpu"})


def test_resolve_device_logic(monkeypatch):
    import pytest
    import torch

    from src.training.train import resolve_device
    from src.utils import config as C

    monkeypatch.setattr(C, "DEVICE", "cpu")
    assert resolve_device() == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(C, "DEVICE", "auto")
    assert resolve_device() == "cpu"                                   # falls back quietly
    monkeypatch.setattr(C, "DEVICE", "cuda")
    with pytest.raises(SystemExit) as e:
        resolve_device()
    assert "cannot use the GPU" in str(e.value)                        # asks loudly instead of silently running on the CPU
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device() == "cuda"
    monkeypatch.setattr(C, "DEVICE", "auto")
    assert resolve_device() == "cuda"


def test_the_default_run_tag_follows_the_start_profile_so_runs_do_not_overwrite_each_other():
    assert build_run({})["name"] == "custom_s0_app"
    assert build_run({"start_profile": "dual"})["name"] == "custom_s0_dual" and build_run({"start_profile": "pair"})["name"] == "custom_s0_pair"
    assert "--tag" in build_run({"start_profile": "pair"})["argv"] and build_run({"start_profile": "pair"})["argv"][-1] == "_pair"
    assert build_run({"start_profile": "pair", "tag": "_mine"})["name"] == "custom_s0_mine"              # a tag you typed is never changed


def test_no_line_comment_in_the_page_swallows_code():
    """A `// comment` put in the middle of a one-line statement silently comments out the rest of that line (it once stopped every episode from loading)."""
    import re
    from pathlib import Path
    lines = (Path(__file__).resolve().parents[1] / "src" / "app" / "index.html").read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("<script"))
    bad = []
    for n, l in enumerate(lines[start:], start=start + 1):
        s = re.sub(r"'(?:\.|[^'\])*'|\"(?:\.|[^\"\])*\"|`(?:\.|[^`\])*`|/(?:\.|[^/\\n])+/[gimsuy]*(?=[.,;)\s])", lambda m: "x" * len(m.group(0)), l)
        i = s.find("//")
        if i >= 0 and re.search(r"[;}]\s*$", l[i:]):               # the comment text itself ends like a statement: code was probably swallowed
            bad.append((n, l[i:i + 120]))
    assert not bad, bad
