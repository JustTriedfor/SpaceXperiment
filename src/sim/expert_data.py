"""Expert demonstrations for behaviour cloning.   python -m src.sim.expert_data --episodes 300 --out expert_sx --make-init sx_init   [--dual]

The scripted expert (src/sim/expert.py) flies the real environment; every step is logged exactly like the trainer's trajectory log
(observation the policy would see, action, scaled reward), so src/training/finetune.py can clone it. Only good landings are kept
(survived and near the pad). To teach recovery and not just the perfect path, the *executed* action is the expert's action plus
Gaussian noise while the *logged* action stays the clean one (the DART trick): the policy then sees off-nominal states with the
right correction attached.

--pair: Dual land with TWO networks: the same pairs of experts, but every lander's own 40-input view and 5 actions are logged to its own data set (<out>_A and <out>_B), one per network.
--dual: Dual land. One expert per lander (they start on the same path, so they land together); the logged observation is the 40-input
dual one and the action the 10-output one. A lander that is already down is labelled "engine off". Only pairs that both landed within
the landing-together tolerance are kept.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from src.sim import vehicle as V
from src.sim.env import SpaceXEnv
from src.sim.expert import Expert
from src.training.data import TrajectoryLogger
from src.utils import config as C

OFF = np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)              # "engine off, everything centred": the label of a lander that is already down


def command_to_action(cmd) -> np.ndarray:
    """The expert's (throttle, gimbal rad, fin rad, rcs_a, rcs_b) -> the environment's 5 actions in [-1, 1]."""
    thr, gim, fin, ra, rb = cmd
    a0 = -1.0 if thr <= 0.0 else -0.5 + 1.5 * (min(1.0, max(V.THROTTLE_MIN, thr)) - V.THROTTLE_MIN) / (1.0 - V.THROTTLE_MIN)
    return np.clip(np.array([a0, gim / V.GIMBAL_MAX, fin / V.FIN_MAX, ra, rb], dtype=np.float32), -1.0, 1.0)


def _label(ex: Expert, sub: SpaceXEnv, down: bool) -> np.ndarray:
    if down:
        return OFF.copy()
    s = sub.s
    return command_to_action(ex.act(s, sub.wind.at(s.t, s.y)))


def _noisy(clean5: np.ndarray, rng, noise: float, down: bool) -> np.ndarray:
    """The executed action: the clean one plus noise, never lighting the engine early and never cutting it during the burn."""
    if down or noise <= 0:
        return clean5
    jitter = rng.normal(0.0, noise, 5).astype(np.float32)
    if clean5[0] <= -0.5:
        jitter[0] = 0.0                                      # never light the engine by accident before the burn
    run = np.clip(clean5 + jitter, -1.0, 1.0)
    if clean5[0] > -0.5:
        run[0] = max(run[0], -0.45)                          # and never cut it by accident during the burn
    return run


def collect(episodes: int, out: str, levels=(0, 0, 1), noise: float = 0.25, seed: int = 0, max_miss_m: float = 80.0, log=print, dual: bool = False, pair: bool = False) -> dict:
    if dual or pair:
        return _collect_dual(episodes, out, levels, noise, seed, max_miss_m, log, pair)
    logger = TrajectoryLogger(C.LOG_DIR / out, out, every=1, flush_episodes=25)
    rng = np.random.default_rng(seed)
    env = SpaceXEnv(level=0, fuel_budget=C.FUEL_BUDGET)
    env.set_rules_stage(2)
    kept = tried = 0
    stats = []
    while kept < episodes and tried < episodes * 6:
        level = int(rng.choice(levels))
        env.set_level(level)
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        ex = Expert()
        ex.reset()
        done = False
        while not done:
            clean = _label(ex, env, False)
            new_obs, r, term, trunc, info = env.step(_noisy(clean, rng, noise, False))
            logger.step(0, obs, clean, r)
            obs, done = new_obs, term or trunc
        m = info["episode_metrics"]
        tried += 1
        good = m["end"] == "rest" and m["dist_m"] <= max_miss_m and not m["out_of_fuel"]
        if good:
            logger.end_episode(0, m, int(m["steps"]))
            kept += 1
            stats.append((m["speed_mps"], m["dist_m"], m["fuel_used"], m["success"]))
        else:
            logger.buf.pop(0, None)
        if tried % 25 == 0:
            log(f"[expert] {kept} kept / {tried} flown")
    logger.flush()
    a = np.array(stats) if stats else np.zeros((1, 4))
    res = dict(kept=kept, tried=tried, median_speed=float(np.median(a[:, 0])), median_miss=float(np.median(a[:, 1])), median_fuel_kg=float(np.median(a[:, 2])),
               successes=int(a[:, 3].sum()))
    log(f"[expert] done: {res}")
    return res


def _loggers(out: str, pair: bool) -> list:
    names = [f"{out}_A", f"{out}_B"] if pair else [out]
    return [TrajectoryLogger(C.LOG_DIR / n, n, every=1, flush_episodes=25) for n in names]


def _collect_dual(episodes, out, levels, noise, seed, max_miss_m, log, pair: bool = False) -> dict:
    from src.sim.dual_env import DualEnv, agent_view
    loggers = _loggers(out, pair)
    rng = np.random.default_rng(seed)
    env = DualEnv(level=0, fuel_budget=C.FUEL_BUDGET)
    env.set_rules_stage(2)
    kept = tried = 0
    stats = []
    while kept < episodes and tried < episodes * 6:
        env.set_level(int(rng.choice(levels)))
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        exs = [Expert(), Expert()]
        for e in exs:
            e.reset()
        done = False
        while not done:
            subs = (env.a, env.b)
            clean = np.concatenate([_label(exs[k], subs[k], env.done[k]) for k in (0, 1)])
            run = np.concatenate([_noisy(clean[5 * k:5 * k + 5], rng, noise, env.done[k]) for k in (0, 1)])
            new_obs, r, term, trunc, info = env.step(run)
            if pair:                                                       # each network gets its own lander's view, its own actions and its own reward
                for k in (0, 1):
                    loggers[k].step(0, agent_view(obs, k), clean[5 * k:5 * k + 5], info["agent_rewards"][k])
            else:
                loggers[0].step(0, obs, clean, r)
            obs, done = new_obs, term or trunc
        m = info["episode_metrics"]
        tried += 1
        good = m["end"] == "rest" and m["dist_m"] <= max_miss_m and not m["out_of_fuel"] and m["sync_dt_s"] == m["sync_dt_s"] and m["sync_dt_s"] <= C.DUAL_SYNC_TOL_S
        if good:
            for lg in loggers:
                lg.end_episode(0, m, int(m["steps"]))
            kept += 1
            stats.append((m["speed_mps"], m["dist_m"], m["fuel_used"], m["success"], m["sync_dt_s"]))
        else:
            for lg in loggers:
                lg.buf.pop(0, None)
        if tried % 25 == 0:
            log(f"[expert] {kept} kept / {tried} flown (pairs)")
    for lg in loggers:
        lg.flush()
    a = np.array(stats) if stats else np.zeros((1, 5))
    res = dict(kept=kept, tried=tried, median_speed=float(np.median(a[:, 0])), median_miss=float(np.median(a[:, 1])), median_fuel_kg=float(np.median(a[:, 2])),
               successes=int(a[:, 3].sum()), median_gap_s=float(np.median(a[:, 4])))
    log(f"[expert] done: {res}")
    return res


def dagger_collect(model_name: str, episodes: int, out: str, levels=(0, 0, 1), noise: float = 0.1, seed: int = 0, log=print) -> dict:
    """DAgger round: the CLONED POLICY flies (with a little noise) and the expert labels every state it visits with the action it would have
    taken. Crashed episodes are kept too: the labels on the way down are exactly the corrections the policy lacks. Single lander, Dual land
    (one network, labelled by one expert per lander) or a PAIR of networks (<name>_A and <name>_B: each flies its own lander and its own
    data set <out>_A / <out>_B is labelled) is detected from the checkpoint name."""
    from stable_baselines3 import PPO
    from src.training.pair import load_pair_models, resolve_pair
    from src.training.weights_io import resolve_checkpoint
    pair = resolve_pair(model_name) is not None
    if pair:
        from src.sim.dual_env import DualEnv, agent_view
        mA, mB = load_pair_models(model_name)
        dual = True
    else:
        model = PPO.load(str(resolve_checkpoint(model_name)), device="cpu")
        dual = int(model.observation_space.shape[0]) == C.DUAL_OBS_DIM
    loggers = _loggers(out, pair)
    rng = np.random.default_rng(seed)
    if dual:
        from src.sim.dual_env import DualEnv
        env = DualEnv(level=0, fuel_budget=C.FUEL_BUDGET)
    else:
        env = SpaceXEnv(level=0, fuel_budget=C.FUEL_BUDGET)
    env.set_rules_stage(2)
    ends: dict = {}
    n_act = C.DUAL_ACT_DIM if dual else 5
    for ep in range(episodes):
        env.set_level(int(rng.choice(levels)))
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        exs = [Expert(), Expert()] if dual else [Expert()]
        for e in exs:
            e.reset()
        done = False
        while not done:
            subs = (env.a, env.b) if dual else (env,)
            down = env.done if dual else [False]
            label = np.concatenate([_label(exs[k], subs[k], down[k]) for k in range(len(subs))])
            if pair:
                act = np.concatenate([mA.predict(agent_view(obs, 0), deterministic=True)[0], mB.predict(agent_view(obs, 1), deterministic=True)[0]])
            else:
                act, _ = model.predict(obs, deterministic=True)
            noise_v = rng.normal(0.0, noise, n_act).astype(np.float32)
            for k in range(len(subs)):
                if down[k]:
                    noise_v[5 * k:5 * k + 5] = 0.0
            run = np.clip(act + noise_v, -1.0, 1.0)
            new_obs, r, term, trunc, info = env.step(run)
            if pair:
                for k in (0, 1):
                    loggers[k].step(0, agent_view(obs, k), label[5 * k:5 * k + 5], info["agent_rewards"][k])
            else:
                loggers[0].step(0, obs, label, r)
            obs, done = new_obs, term or trunc
        m = info["episode_metrics"]
        ends[m["end"]] = ends.get(m["end"], 0) + 1
        for lg in loggers:
            lg.end_episode(0, m, int(m["steps"]))
    for lg in loggers:
        lg.flush()
    log(f"[dagger] {episodes} episodes flown by {model_name}: {ends}")
    return ends


def make_init(name: str, dual: bool = False, from_single: str | None = None, pair: bool = False) -> None:
    """Save a freshly initialised network as a checkpoint to clone into: one lander (256 x 256), or Dual land (512 x 512, 40 -> 10).
    With from_single (dual only) the dual network starts as two copies of that single-lander checkpoint instead of random weights."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(1)
    C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if pair:                                                                   # two networks <name>_A and <name>_B
        from src.training.pair import make_pair, new_agent_model
        if from_single:
            make_pair(from_single, name)
            print(f"saved the pair {name} (_A and _B): two copies of {from_single}")
        else:
            for k, s in enumerate("AB"):
                new_agent_model(seed=k).save(str(C.CHECKPOINT_DIR / f"{name}_{s}"))
            print(f"saved an untrained pair {name} (_A and _B)")
        return
    if dual:
        from src.sim.dual_init import make_dual, new_dual_model
        if from_single:
            r = make_dual(from_single, name)
            print(f"saved {r['checkpoint']}: two copies of {from_single} side by side")
            return
        new_dual_model().save(str(C.CHECKPOINT_DIR / name))
        print(f"saved an untrained dual network ({C.NET_ARCH_DUAL}, {C.DUAL_OBS_DIM} inputs, {C.DUAL_ACT_DIM} outputs) as {name}.zip")
        return
    arch = dict(pi=list(C.NET_ARCH), vf=list(C.NET_ARCH))
    model = PPO("MlpPolicy", DummyVecEnv([lambda: SpaceXEnv()]), seed=0, device="cpu", verbose=0, policy_kwargs=dict(net_arch=arch), **C.PPO_KWARGS)
    model.save(str(C.CHECKPOINT_DIR / name))
    print(f"saved an untrained {C.NET_ARCH} network as {name}.zip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", default="expert_sx")
    ap.add_argument("--noise", type=float, default=0.25)
    ap.add_argument("--levels", default="0,0,1", help="comma list; repeats weight a level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pair", action="store_true", help="Dual land with TWO networks: log each lander's own view to <out>_A and <out>_B")
    ap.add_argument("--dual", action="store_true", help="Dual land: fly two landers per episode (one expert each) and log the 40-input / 10-output data")
    ap.add_argument("--make-init", default=None, help="also save an untrained network under this name")
    ap.add_argument("--from-single", default=None, help="with --dual and --make-init: start the dual network as two copies of this single-lander checkpoint")
    a = ap.parse_args()
    collect(a.episodes, a.out, tuple(int(x) for x in a.levels.split(",")), a.noise, a.seed, dual=a.dual, pair=a.pair)
    if a.make_init:
        make_init(a.make_init, dual=a.dual, from_single=a.from_single, pair=a.pair)


if __name__ == "__main__":
    main()
