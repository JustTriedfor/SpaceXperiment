"""Two neural networks in one environment (start profile "pair"): Dual land with one network PER LANDER.

The environment is the same DualEnv (two landers, two pads, one shared wind, landing-together reward), but nothing is shared between the
landers' brains: lander A is flown by network A, lander B by network B, each a normal 256 x 256 PPO policy with its own weights, its own
critic and its own optimiser, learning from its own reward (its own landing score + the shared landing-together terms).

What each network sees (40 numbers): its OWN lander's 18 observations first, then its partner's 18, then [height gap, fall-speed gap,
own down, partner down] (src.sim.dual_env.agent_view). What it writes: its own 5 actions.

How the two are trained together with stock Stable-Baselines3: every network is an ordinary PPO model whose environment is an `AgentView`;
the two views share one `PairVecEnv`. The two models learn in two threads; each time both have chosen their actions the shared environment
takes one step (a barrier), and each view hands its network its own observations, rewards and episode ends. So both collect the same steps,
update at the same time, and nothing in PPO had to be rewritten.

Checkpoints: a pair is two files, <name>_A.zip and <name>_B.zip; the apps and scripts take the base name <name>.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from src.sim.dual_env import DualEnv, agent_view
from src.utils import config as C

OBS = C.DUAL_OBS_DIM                                   # 40: what each network sees
ACT = 5                                                # what each network writes


# ---------------------------------------------------------------- checkpoints of a pair
def pair_files(name: str) -> tuple:
    return C.CHECKPOINT_DIR / f"{name}_A.zip", C.CHECKPOINT_DIR / f"{name}_B.zip"


def resolve_pair(name):
    """'mypair', 'mypair_A' or 'mypair_B' -> ('mypair', pathA, pathB) if both networks exist, else None."""
    n = Path(str(name)).stem if str(name).endswith(".zip") else str(name)
    base = n[:-2] if n.endswith(("_A", "_B")) else n
    a, b = pair_files(base)
    return (base, a, b) if a.exists() and b.exists() else None


class _AgentStub(gym.Env):
    """Only carries the observation and action spaces of one network (to build or load a model without a real environment)."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-20.0, 20.0, (OBS,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(OBS, np.float32), {}

    def step(self, action):
        return np.zeros(OBS, np.float32), 0.0, True, False, {}


def new_agent_model(seed: int = 0, env=None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    arch = dict(pi=list(C.NET_ARCH), vf=list(C.NET_ARCH))
    return PPO("MlpPolicy", env if env is not None else DummyVecEnv([_AgentStub]), seed=seed, device="cpu", verbose=0, policy_kwargs=dict(net_arch=arch), **C.PPO_KWARGS)


def copy_single_into_agent(single, agent) -> None:
    """The agent network starts as the single-lander network: its weights on the first 18 inputs (the lander's own observation), zero on the
    22 that describe the partner, so it flies exactly like the single-lander one until PPO teaches it to use them."""
    import torch
    sp, ap = single.policy, agent.policy
    with torch.no_grad():
        for net in ("policy_net", "value_net"):
            s, a = getattr(sp.mlp_extractor, net), getattr(ap.mlp_extractor, net)
            a[0].weight.zero_()
            a[0].weight[:, :s[0].in_features] = s[0].weight
            a[0].bias.copy_(s[0].bias)
            a[2].weight.copy_(s[2].weight)
            a[2].bias.copy_(s[2].bias)
        ap.action_net.weight.copy_(sp.action_net.weight)
        ap.action_net.bias.copy_(sp.action_net.bias)
        ap.value_net.weight.copy_(sp.value_net.weight)
        ap.value_net.bias.copy_(sp.value_net.bias)
        ap.log_std.copy_(sp.log_std)


def make_pair(src_name: str, out_name: str) -> dict:
    """Two network files <out>_A.zip and <out>_B.zip, each a copy of the single-lander checkpoint (never overwrites)."""
    from stable_baselines3 import PPO
    from src.training.weights_io import resolve_checkpoint
    a, b = pair_files(out_name)
    if a.exists() or b.exists():
        raise ValueError(f"{out_name} already exists (a pair is {out_name}_A and {out_name}_B); pick another name")
    single = PPO.load(str(resolve_checkpoint(src_name)), device="cpu")
    if (single.observation_space.shape[0], single.action_space.shape[0]) != (18, 5):
        raise ValueError(f"{src_name} is not a single-lander checkpoint ({single.observation_space.shape[0]} inputs, {single.action_space.shape[0]} outputs)")
    C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for k, path in enumerate((a, b)):
        agent = new_agent_model(seed=k)
        copy_single_into_agent(single, agent)
        agent.save(str(path.with_suffix("")))
    return dict(checkpoint=out_name, files=[a.name, b.name], source=src_name)


# ---------------------------------------------------------------- one environment, two views
class PairVecEnv:
    """The real vector environment (of DualEnv) shared by the two views. It steps when BOTH networks have submitted their actions."""

    def __init__(self, vec, record_activations: bool = False):
        self.vec, self.n = vec, vec.num_envs
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(2, action=self._step_both, timeout=3600)
        self.acts = [None, None]
        self.res = None
        self.joint = None
        self.fresh = [False, False]
        self.record = record_activations
        self.bufs = [[[] for _ in range(self.n)] for _ in range(2)]      # per network, per env: the (observation, action) pairs of the running episode

    def reset(self, k: int):
        with self.lock:
            if self.fresh[k]:
                self.fresh[k] = False                                    # the partner already reset for both
            else:
                self.joint = self.vec.reset()
                self.fresh[1 - k] = True
                self.bufs = [[[] for _ in range(self.n)] for _ in range(2)]
        return agent_view(self.joint, k)

    def submit(self, k: int, actions) -> None:
        self.acts[k] = np.asarray(actions, dtype=np.float32).reshape(self.n, ACT)
        self.barrier.wait()

    def _step_both(self) -> None:                                         # runs once per step, in whichever thread arrived last
        if self.record:
            for k in (0, 1):
                view = agent_view(self.joint, k)
                for i in range(self.n):
                    self.bufs[k][i].append((view[i].copy(), self.acts[k][i].copy()))
        self.res = self.vec.step(np.concatenate(self.acts, axis=1))
        self.joint = self.res[0]

    def abort(self) -> None:
        self.barrier.abort()


class AgentView(VecEnv):
    """What ONE network sees of the shared environment: its observations (own lander first), its own reward and the shared episode ends."""

    def __init__(self, pair: PairVecEnv, k: int):
        super().__init__(pair.n, spaces.Box(-20.0, 20.0, (OBS,), dtype=np.float32), spaces.Box(-1.0, 1.0, (ACT,), dtype=np.float32))
        self.pair, self.k = pair, k

    def reset(self):
        return self.pair.reset(self.k)

    def step_async(self, actions) -> None:
        self.pair.submit(self.k, actions)

    def step_wait(self):
        obs, _, dones, infos = self.pair.res
        k = self.k
        rewards = np.array([inf["agent_rewards"][k] for inf in infos], dtype=np.float32)
        out = []
        for inf in infos:
            if k == 0:
                d = dict(inf)                                             # network A's infos carry everything (metrics, the recorded episode)
            else:
                d = {key: inf[key] for key in ("TimeLimit.truncated",) if key in inf}
            if "terminal_observation" in inf:
                d["terminal_observation"] = agent_view(inf["terminal_observation"], k)
            out.append(d)
        return agent_view(obs, k), rewards, dones.copy(), out

    def close(self) -> None:
        pass

    def get_attr(self, attr_name, indices=None):
        return self.pair.vec.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        return self.pair.vec.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.pair.vec.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.pair.n

    def seed(self, seed=None):
        return self.pair.vec.seed(seed)


def load_pair_models(name: str, device: str = "cpu"):
    """(model A, model B) from <name>_A.zip / <name>_B.zip (no environment needed: for flying them)."""
    from stable_baselines3 import PPO
    r = resolve_pair(name)
    if r is None:
        raise FileNotFoundError(f"no pair named {name} (needs {name}_A.zip and {name}_B.zip)")
    return PPO.load(str(r[1]), device=device), PPO.load(str(r[2]), device=device)


# ---------------------------------------------------------------- training
def run_pair(args, name: str, run_dir: Path, ppo_kwargs: dict, CurriculumCallback, device: str = "cpu") -> None:
    """Called by train.py when the start profile is 'pair'. Same run folder, curriculum, live stream and logs as a normal run; two networks."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from src.training.brain import episode_payload
    from src.training.env import make_vec_env

    vec = make_vec_env(args.n_envs, args.seed, record_all=not args.no_live, level=args.start_level, reward=args.reward)
    pair = PairVecEnv(vec, record_activations=bool(C.LIVE_ACTIVATIONS and not args.no_live))
    views = [AgentView(pair, 0), AgentView(pair, 1)]
    tb = None
    try:
        import tensorboard  # noqa: F401
        tb = str(run_dir / "tb")
    except ImportError:
        pass
    arch = dict(pi=list(C.NET_ARCH), vf=list(C.NET_ARCH))
    models = [PPO("MlpPolicy", views[k], seed=args.seed + k, device=device, verbose=0, tensorboard_log=(f"{tb}_{'AB'[k]}" if tb else None),
                  policy_kwargs=dict(net_arch=arch), **ppo_kwargs) for k in (0, 1)]
    if args.init_from:
        _init_pair(models, args.init_from)
    (C.MODEL_CONFIG_DIR / f"{name}.json").write_text(json.dumps(
        dict(vars(args), ppo=ppo_kwargs, fuel_budget=C.FUEL_BUDGET, levels=C.LEVELS, overrides=C.OVERRIDES, networks=2, profile="pair"), indent=2))

    class Lead(CurriculumCallback):
        """The normal curriculum / logging / live-stream callback, driven by network A; it also saves and streams network B."""

        def _save_level(self, path):
            models[0].save(f"{path}_A")
            models[1].save(f"{path}_B")

        def _extra_payload(self, traj, buf_idx):
            buf = pair.bufs[1][buf_idx]
            if C.LIVE_ACTIVATIONS and buf:
                try:
                    traj["actB"] = episode_payload(models[1].policy, np.array([o for o, _ in buf]), np.array([a for _, a in buf]))
                except Exception as e:
                    print(f"[brain] network B activations skipped: {type(e).__name__}: {e}", flush=True)

        def _on_step(self) -> bool:
            ok = super()._on_step()
            for i, info in enumerate(self.locals["infos"]):
                if "episode_metrics" in info:
                    pair.bufs[0][i], pair.bufs[1][i] = [], []
            return ok

    class Follow(BaseCallback):
        def _on_step(self) -> bool:
            return True

    lead = Lead(run_dir, C.CHECKPOINT_DIR / name, args.start_level, args.stop_at_level, args.snapshot_every, live=not args.no_live,
                live_dir=args.live_dir, live_env=args.live_env)
    cbs = [lead, Follow()]
    errors: list = []

    def work(k: int):
        try:
            models[k].learn(args.total_steps, callback=cbs[k], tb_log_name=f"ppo_{'AB'[k]}")
        except threading.BrokenBarrierError:
            pass                                                           # the other network finished or stopped first
        except BaseException as e:                                         # one network failing must not leave the other waiting forever
            errors.append(e)
        finally:
            pair.abort()

    ts = [threading.Thread(target=work, args=(k,), daemon=True) for k in (0, 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errors:
        raise errors[0]
    for k in (0, 1):
        models[k].save(C.CHECKPOINT_DIR / f"{name}_{'AB'[k]}")
    vec.close()
    print(f"done: {args.total_steps} steps requested, ran {models[0].num_timesteps} (two networks: {name}_A and {name}_B)", flush=True)


def _init_pair(models, init_from: str) -> None:
    """--init-from NAME: a pair (NAME_A / NAME_B), or a single-lander checkpoint (each network starts as a copy of it)."""
    from stable_baselines3 import PPO
    from src.training.weights_io import resolve_checkpoint
    r = resolve_pair(init_from)
    if r is not None:
        for k, path in enumerate((r[1], r[2])):
            models[k].set_parameters(str(path), exact_match=False)
        print(f"[init] weights loaded from the pair {r[0]} (A and B)", flush=True)
        return
    src = resolve_checkpoint(init_from)
    single = PPO.load(str(src), device="cpu")
    if (single.observation_space.shape[0], single.action_space.shape[0]) != (18, 5):
        sys.exit(f"Cannot start from {src.name}: a two-network run needs a pair ({init_from}_A and {init_from}_B) or a single-lander checkpoint "
                 f"(each network then starts as a copy of it); this one has {single.observation_space.shape[0]} inputs and {single.action_space.shape[0]} outputs.")
    for m in models:
        copy_single_into_agent(single, m)
    print(f"[init] both networks start as copies of the single-lander checkpoint {src.name} (weights on the 18 own inputs, zero on the partner inputs)", flush=True)
