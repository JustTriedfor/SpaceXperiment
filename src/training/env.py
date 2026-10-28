"""Environment factory: our own 2-D landing simulator (src/sim), one environment per worker."""
from __future__ import annotations

from functools import partial

import gymnasium as gym

from src.sim.env import SpaceXEnv
from src.utils import config as C


def make_env(level: int = 0, reward: str = "custom", fuel_budget: float | None = None, render_mode: str | None = None, randomize: bool = True,
             record: bool = False) -> gym.Env:
    """The booster landing environment. The rules (tilt limit, ignition limit, clock, stall, fuel) and the custom reward live inside it;
    `reward`, `render_mode` and `randomize` are kept only so old call sites keep working (there is no second reward type any more)."""
    if reward not in ("custom", "default"):
        raise ValueError(f"unknown reward type: {reward}")
    if C.START_PROFILE in ("dual", "pair"):                              # Dual land: one environment, two landers, two pads
        from src.sim.dual_env import DualEnv
        return DualEnv(level=level, fuel_budget=fuel_budget, record=record)
    return SpaceXEnv(level=level, fuel_budget=fuel_budget, record=record)


def make_vec_env(n_envs: int, seed: int, record_all: bool = False, **kwargs):
    """SubprocVecEnv factory. Call only from inside `if __name__ == "__main__":` on Windows."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = [partial(make_env, record=record_all, **kwargs) for _ in range(n_envs)]
    # one env: run in-process (saves a worker process per agent in learning-rate sweeps)
    vec = DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns)
    vec.seed(seed)
    return vec
