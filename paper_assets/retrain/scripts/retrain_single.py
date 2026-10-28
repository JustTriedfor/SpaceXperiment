"""Retrain the single-lander controller (expert demonstrations -> behaviour cloning -> DAgger rounds) with explicit, disjoint seed ranges.

    python paper_assets/retrain/scripts/retrain_single.py --name rt1 --config paper_assets/retrain/attempt1.json

Nothing in src/ is changed: it only imports the project's Expert, SpaceXEnv, TrajectoryLogger and finetune().

SEEDS (env.reset(seed=...); every level uses the same ranges)
    TEST        0 .. 299          never used here, not even for choosing: paper_assets/scripts/run_eval.py reads them once at the very end
    VALIDATION  100000 .. 100000+val_episodes-1   used to choose the checkpoint (after every DAgger round) and to decide whether to try another recipe
    TRAINING    expert demonstrations 1,000,000+i ; DAgger round r: 2,000,000 + r * 100,000 + i      (the run asserts the ranges do not overlap)

The dataset names are <name>_expert_w<k> and <name>_dagger_<r>_w<k> (one logger per worker process; finetune() reads the lists). Checkpoints: <name>_clone,
<name>_dagger_<r>, and <name>_final = the DAGGER ROUND (or the clone) WITH THE BEST VALIDATION SCORE (mean success over all four levels on the validation seeds).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

TEST_RANGE = (0, 300)
VAL0 = 100_000
EXPERT_SEED0 = 1_000_000
DAGGER_SEED0 = 2_000_000
RETRAIN = ROOT / "paper_assets" / "retrain"


def _init():
    import torch
    torch.set_num_threads(1)


def expert_worker(job):
    """Fly the scripted expert on the given (seed, level) pairs; keep the good landings exactly as src.sim.expert_data.collect does (noise on the executed action, clean label)."""
    out, pairs, noise, max_miss_m = job
    from src.sim.env import SpaceXEnv
    from src.sim.expert import Expert
    from src.sim.expert_data import _label, _noisy
    from src.training.data import TrajectoryLogger
    from src.utils import config as C
    logger = TrajectoryLogger(C.LOG_DIR / out, out, every=1, flush_episodes=25)
    env = SpaceXEnv(level=0, fuel_budget=C.FUEL_BUDGET)
    env.set_rules_stage(2)
    kept = succ = 0
    for seed, level in pairs:
        rng = np.random.default_rng(seed)
        env.set_level(level)
        obs, _ = env.reset(seed=seed)
        ex = Expert()
        ex.reset()
        done = False
        while not done:
            clean = _label(ex, env, False)
            new_obs, r, term, trunc, info = env.step(_noisy(clean, rng, noise, False))
            logger.step(0, obs, clean, r)
            obs, done = new_obs, term or trunc
        m = info["episode_metrics"]
        if m["end"] == "rest" and m["dist_m"] <= max_miss_m and not m["out_of_fuel"]:
            logger.end_episode(0, m, int(m["steps"]))
            kept += 1
            succ += int(bool(m["success"]))
        else:
            logger.buf.pop(0, None)
    logger.flush()
    return len(pairs), kept, succ


def dagger_worker(job):
    """The cloned policy flies (a little noise on the executed action) and the expert labels every state it visits; crashed episodes are kept, as in dagger_collect."""
    model_name, out, pairs, noise = job
    from stable_baselines3 import PPO
    from src.sim.env import SpaceXEnv
    from src.sim.expert import Expert
    from src.sim.expert_data import _label
    from src.training.data import TrajectoryLogger
    from src.training.weights_io import resolve_checkpoint
    from src.utils import config as C
    model = PPO.load(str(resolve_checkpoint(model_name)), device="cpu")
    logger = TrajectoryLogger(C.LOG_DIR / out, out, every=1, flush_episodes=25)
    env = SpaceXEnv(level=0, fuel_budget=C.FUEL_BUDGET)
    env.set_rules_stage(2)
    n = succ = 0
    for seed, level in pairs:
        rng = np.random.default_rng(seed)
        env.set_level(level)
        obs, _ = env.reset(seed=seed)
        ex = Expert()
        ex.reset()
        done = False
        while not done:
            label = _label(ex, env, False)
            act, _ = model.predict(obs, deterministic=True)
            run = np.clip(act + rng.normal(0.0, noise, 5).astype(np.float32), -1.0, 1.0)
            new_obs, r, term, trunc, info = env.step(run)
            logger.step(0, obs, label, r)
            obs, done = new_obs, term or trunc
        m = info["episode_metrics"]
        logger.end_episode(0, m, int(m["steps"]))
        n += 1
        succ += int(bool(m["success"]))
    logger.flush()
    return n, succ


def fit(checkpoint, sources, out_name, say, tag, **kw):
    """finetune() in a child process (the GPU context stays out of this process, whose worker pools need the memory)."""
    args = dict(checkpoint=checkpoint, sources=sources, out_name=out_name, **kw)
    p = subprocess.run([sys.executable, str(Path(__file__).with_name("fit_worker.py")), json.dumps(args)], capture_output=True, text=True, cwd=ROOT)
    if p.returncode:
        raise RuntimeError("fit failed: " + (p.stderr or p.stdout)[-1500:])
    for line in p.stdout.splitlines():
        if "[data]" in line or "[done]" in line or "epoch " + str(kw["epochs"]) + "/" in line:
            say(f"[{tag}] " + line)


def split(pairs, k):
    return [pairs[i::k] for i in range(k)]


def validate(ckpt: str, cfg: dict) -> dict:
    """Run paper_assets/scripts/run_eval.py (the same evaluator as the paper) on the validation seeds; returns success rate per level and the mean."""
    out = RETRAIN / "val" / f"{ckpt}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([sys.executable, str(ROOT / "paper_assets" / "scripts" / "run_eval.py"), "--controllers", ckpt, "--levels", "0,1,2,3", "--episodes", str(cfg["val_episodes"]),
                    "--seed0", str(VAL0), "--workers", str(cfg["workers"]), "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    if p.returncode:
        raise RuntimeError("validation run failed: " + (p.stderr or p.stdout)[-1500:])
    import csv
    rows = list(csv.DictReader(open(out)))
    rate = {lv: float(np.mean([int(r["success"]) for r in rows if int(r["level"]) == lv])) for lv in range(4)}
    return dict(levels=rate, mean=float(np.mean(list(rate.values()))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = dict(start="sx_imitation_quiet", expert_episodes=800, expert_levels="0,1,1,2,2,3", expert_noise=0.12, clone_epochs=40, clone_lr=1e-3, rounds=8, dagger_episodes=180,
               dagger_levels="0,1,1,2,2,3,3", dagger_noise=0.1, dagger_epochs=20, dagger_lr=5e-4, final_lr=1e-4, final_rounds=2, batch=1024, low_weight=5.0, log_std=-1.6,
               val_episodes=200, workers=12, device="cuda", net_arch=None, seed=0)
    cfg.update(json.loads(Path(a.config).read_text()))
    name = a.name
    t00 = time.time()
    rng = np.random.default_rng(cfg["seed"])
    elv = [int(x) for x in str(cfg["expert_levels"]).split(",")]
    dlv = [int(x) for x in str(cfg["dagger_levels"]).split(",")]
    n_exp_try = int(cfg["expert_episodes"] * 1.12)
    exp_pairs = [(EXPERT_SEED0 + i, int(rng.choice(elv))) for i in range(n_exp_try)]
    val_hi = VAL0 + cfg["val_episodes"] - 1
    assert TEST_RANGE[1] <= VAL0 and val_hi < EXPERT_SEED0 and EXPERT_SEED0 + n_exp_try < DAGGER_SEED0, "seed ranges overlap"
    log = RETRAIN / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    def say(s):
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - t00:5.0f}s] {s}"
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(line + "\n")

    say(f"attempt {name}: {json.dumps(cfg)}")
    say(f"seeds: test {TEST_RANGE[0]}-{TEST_RANGE[1] - 1} (not used), validation {VAL0}-{val_hi}, expert training {EXPERT_SEED0}-{EXPERT_SEED0 + n_exp_try - 1}, DAgger round r {DAGGER_SEED0}+r*100000+i")
    workers = int(cfg["workers"])
    # ---- starting network
    from src.sim.expert_data import make_init
    from src.utils import config as C
    start = cfg["start"]
    if cfg["net_arch"]:                                                    # a different network size: a fresh, untrained network of that size
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from src.sim.env import SpaceXEnv
        m0 = PPO("MlpPolicy", DummyVecEnv([lambda: SpaceXEnv()]), seed=cfg["seed"], device="cpu", verbose=0, policy_kwargs=dict(net_arch=list(cfg["net_arch"])), **C.PPO_KWARGS)
        m0.save(str(C.CHECKPOINT_DIR / f"{name}_init"))
        start = f"{name}_init"
        say(f"fresh network {cfg['net_arch']}")
    # ---- 1. expert demonstrations (parallel)
    with Pool(workers, initializer=_init) as pool:
        res = pool.map(expert_worker, [(f"{name}_expert_w{k}", p, cfg["expert_noise"], 80.0) for k, p in enumerate(split(exp_pairs, workers))])
    flown, kept, succ = (sum(r[i] for r in res) for i in range(3))
    say(f"expert: {kept} landings kept of {flown} flown ({succ} of them strict successes)")
    exp_sets = [f"{name}_expert_w{k}" for k in range(workers)]
    # ---- 2. clone
    fit(start, exp_sets, f"{name}_clone", say, "clone", epochs=cfg["clone_epochs"], lr=cfg["clone_lr"], batch=cfg["batch"], log_std=cfg["log_std"], low_weight=cfg["low_weight"],
        device=cfg["device"], seed=cfg["seed"])
    rounds = [dict(round=0, checkpoint=f"{name}_clone", val=validate(f"{name}_clone", cfg))]
    say(f"validation {name}_clone: {json.dumps(rounds[0]['val'])}")
    data = list(exp_sets)
    cur = f"{name}_clone"
    # ---- 3. DAgger rounds
    for r in range(1, cfg["rounds"] + 1):
        lr = cfg["final_lr"] if r > cfg["rounds"] - cfg["final_rounds"] else cfg["dagger_lr"]
        base = DAGGER_SEED0 + r * 100_000
        pairs = [(base + i, int(rng.choice(dlv))) for i in range(cfg["dagger_episodes"])]
        assert base + len(pairs) < base + 100_000
        with Pool(workers, initializer=_init) as pool:
            res = pool.map(dagger_worker, [(cur, f"{name}_dagger_{r}_w{k}", p, cfg["dagger_noise"]) for k, p in enumerate(split(pairs, workers))])
        data += [f"{name}_dagger_{r}_w{k}" for k in range(workers)]
        say(f"round {r}: {cur} flew {sum(x[0] for x in res)} DAgger episodes, {sum(x[1] for x in res)} strict successes (with noise {cfg['dagger_noise']})")
        out = f"{name}_dagger_{r}"
        fit(cur, data, out, say, "fit", epochs=cfg["dagger_epochs"], lr=lr, batch=cfg["batch"], log_std=cfg["log_std"], low_weight=cfg["low_weight"], device=cfg["device"], seed=cfg["seed"] + r)
        cur = out
        v = validate(out, cfg)
        rounds.append(dict(round=r, checkpoint=out, lr=lr, val=v))
        say(f"validation {out}: {json.dumps(v)}")
    best = max(rounds, key=lambda x: (x["val"]["mean"], x["round"]))
    import shutil
    shutil.copy2(C.CHECKPOINT_DIR / f"{best['checkpoint']}.zip", C.CHECKPOINT_DIR / f"{name}_final.zip")
    say(f"{name}_final = {best['checkpoint']} (validation mean {best['val']['mean']:.4f})")
    (RETRAIN / "logs" / f"{name}_result.json").write_text(json.dumps(dict(config=cfg, rounds=rounds, final=best["checkpoint"], seconds=round(time.time() - t00)), indent=1))
    say(f"done in {time.time() - t00:.0f} s")


if __name__ == "__main__":
    main()
