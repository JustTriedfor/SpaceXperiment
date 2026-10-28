"""Make a dual-lander network from a single-lander one:  python -m src.sim.dual_init --from sx_imitation_quiet --out dual_imitation

The dual policy reads both landers' observations and writes both landers' actions. Its weights are built in blocks so that, to begin with, it
IS two independent copies of the single-lander network, one per lander (twice as wide: 512 x 512). The four extra inputs (height gap, fall-speed
gap, A down, B down) start with zero weights, so PPO then learns how to use them to land together. The critic is the sum of the two single
critics (the dual reward is the sum of two landers' rewards).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.sim.dual_env import DualEnv
from src.sim.env import ACT_DIM, OBS_DIM
from src.utils import config as C


def new_dual_model(seed: int = 0):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    arch = dict(pi=list(C.NET_ARCH_DUAL), vf=list(C.NET_ARCH_DUAL))
    return PPO("MlpPolicy", DummyVecEnv([lambda: DualEnv()]), seed=seed, device="cpu", verbose=0, policy_kwargs=dict(net_arch=arch), **C.PPO_KWARGS)


def _lin(seq, i):
    return seq[i]


def block_copy(single, dual) -> None:
    """Write two copies of `single` (a PPO model) into `dual` (a PPO model built by new_dual_model). Works in place."""
    sp, dp = single.policy, dual.policy
    h = sp.mlp_extractor.policy_net[0].out_features
    assert dp.mlp_extractor.policy_net[0].out_features == 2 * h, "the dual network must be exactly twice as wide as the single one"
    with torch.no_grad():
        for net in ("policy_net", "value_net"):
            s_net, d_net = getattr(sp.mlp_extractor, net), getattr(dp.mlp_extractor, net)
            w1, b1, w2, b2 = s_net[0].weight, s_net[0].bias, s_net[2].weight, s_net[2].bias
            d_net[0].weight.zero_()
            d_net[0].weight[:h, :OBS_DIM] = w1
            d_net[0].weight[h:, OBS_DIM:2 * OBS_DIM] = w1
            d_net[0].bias.copy_(torch.cat([b1, b1]))
            d_net[2].weight.zero_()
            d_net[2].weight[:h, :h] = w2
            d_net[2].weight[h:, h:] = w2
            d_net[2].bias.copy_(torch.cat([b2, b2]))
        dp.action_net.weight.zero_()
        dp.action_net.weight[:ACT_DIM, :h] = sp.action_net.weight
        dp.action_net.weight[ACT_DIM:, h:] = sp.action_net.weight
        dp.action_net.bias.copy_(torch.cat([sp.action_net.bias, sp.action_net.bias]))
        dp.value_net.weight.copy_(torch.cat([sp.value_net.weight, sp.value_net.weight], dim=1))      # V = V_A + V_B
        dp.value_net.bias.copy_(2.0 * sp.value_net.bias)
        dp.log_std.copy_(torch.cat([sp.log_std, sp.log_std]))


def make_dual(src_name: str, out_name: str) -> dict:
    """Writes models/checkpoints/<out_name>.zip: two copies of the single-lander checkpoint side by side. Never overwrites an existing one."""
    from stable_baselines3 import PPO
    from src.training.weights_io import resolve_checkpoint
    dest = C.CHECKPOINT_DIR / f"{out_name}.zip"
    if dest.exists():
        raise ValueError(f"{out_name} already exists; pick another name")
    single = PPO.load(str(resolve_checkpoint(src_name)), device="cpu")
    if (single.observation_space.shape[0], single.action_space.shape[0]) != (OBS_DIM, ACT_DIM):
        raise ValueError(f"{src_name} is not a single-lander checkpoint ({single.observation_space.shape[0]} inputs, {single.action_space.shape[0]} outputs)")
    if single.policy.mlp_extractor.policy_net[0].out_features * 2 != C.NET_ARCH_DUAL[0]:
        raise ValueError(f"{src_name} has {single.policy.mlp_extractor.policy_net[0].out_features} hidden units; the dual network ({C.NET_ARCH_DUAL[0]}) must be exactly twice as wide")
    dual = new_dual_model()
    block_copy(single, dual)
    C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dual.save(str(dest.with_suffix("")))
    return dict(checkpoint=dest.name, source=src_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True, help="single-lander checkpoint name (for example sx_imitation_quiet)")
    ap.add_argument("--out", required=True, help="name of the dual checkpoint to write")
    a = ap.parse_args()
    r = make_dual(a.src, a.out)
    print(f"saved {r['checkpoint']}: two copies of {a.src} side by side ({C.NET_ARCH_DUAL}, {C.DUAL_OBS_DIM} inputs, {C.DUAL_ACT_DIM} outputs)")


if __name__ == "__main__":
    main()
