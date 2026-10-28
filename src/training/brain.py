"""Neural-network 'brain view' data: what the policy network computed at every step of an episode.

The live episodes are recorded inside the simulation workers, which never see the network, so the trainer (where the policy lives)
recomputes the activations for each finished episode from the observations it fed the policy, and attaches them to the episode file.

Network (stable-baselines3 MlpPolicy, default): inputs (11) -> 64 tanh -> 64 tanh -> action mean (2: main engine, side engine).
The value estimate comes from a separate network of the same shape.
Everything is rounded and flattened (row-major, one step after another) to keep the files small.
"""
from __future__ import annotations

import numpy as np
import torch


def _flat(a, nd: int) -> list:
    return np.round(np.asarray(a, dtype=np.float64), nd).ravel().tolist()


def _ints(a, scale: int = 100) -> list:
    return np.round(np.asarray(a, dtype=np.float64) * scale).astype(int).ravel().tolist()


def policy_weights(policy) -> dict:
    """The three weight matrices (out x in, row-major) and biases of the policy network as flat lists, rounded to 3 decimals."""
    pn = policy.mlp_extractor.policy_net
    with torch.no_grad():
        return dict(
            w1=_flat(pn[0].weight.cpu().numpy(), 3),
            w2=_flat(pn[2].weight.cpu().numpy(), 3),
            w3=_flat(policy.action_net.weight.cpu().numpy(), 3),
            b1=_flat(pn[0].bias.cpu().numpy(), 3),               # biases + the observation scales let the browser run the network itself
            b2=_flat(pn[2].bias.cpu().numpy(), 3),               # (the decision-boundary view evaluates it on a grid)
            b3=_flat(policy.action_net.bias.cpu().numpy(), 3),
        )


def network_trace(policy, obs, actions=None) -> dict:
    """Activations for a whole episode in one batch.

    obs: (T, obs_dim) the observations the policy received, one per step.  actions: (T, 2) the actions that were executed.
    Returns flat lists: x (T*obs_dim inputs), h1 and h2 (T*64 each, activations x100 as ints), m (T*2 action mean),
    a (T*2 executed action), v (T value estimates), plus dims.
    """
    obs = np.asarray(obs, dtype=np.float32)
    was_training = policy.training
    policy.set_training_mode(False)
    try:
        with torch.no_grad():
            x = torch.as_tensor(obs, device=policy.device)
            pn = policy.mlp_extractor.policy_net            # Sequential: Linear, Tanh, Linear, Tanh
            h1 = pn[1](pn[0](x))
            h2 = pn[3](pn[2](h1))
            mean = policy.action_net(h2)
            value = policy.value_net(policy.mlp_extractor.value_net(x))
    finally:
        policy.set_training_mode(was_training)
    out = dict(
        dims=[int(obs.shape[1]), int(h1.shape[1]), int(h2.shape[1]), int(mean.shape[1])],
        x=_flat(obs, 3),
        h1=_ints(h1.cpu().numpy()),
        h2=_ints(h2.cpu().numpy()),
        m=_flat(mean.cpu().numpy(), 3),
        v=_flat(value.cpu().numpy(), 2),
    )
    if actions is not None:
        out["a"] = _flat(np.asarray(actions, dtype=np.float32), 3)
    return out


def episode_payload(policy, obs, actions=None) -> dict:
    """Everything the viewer needs for one episode: the trace plus the weights in force when it was recorded."""
    from src.utils import config as C
    scales = dict(y=C.OBS_Y_TO_M, vy=C.OBS_VY_TO_MPS, vx=C.OBS_VX_TO_MPS, x=C.OBS_X_TO_M)
    return {**network_trace(policy, obs, actions), **policy_weights(policy), "scales": scales}
