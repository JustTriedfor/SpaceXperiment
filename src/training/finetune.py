"""Train a checkpoint further from logged experience (behaviour cloning + value warm-up). Never overwrites its source.

    python src/training/finetune.py --checkpoint custom_s0_app --data custom_s0_app --out custom_s0_app_clone \
        [--min-merit 40] [--soft-only] [--min-return R] [--max-return R] [--kept best+sample] [--epochs 30] [--lr 1e-3] [--no-bc] [--no-value]

--data is a run name under data/trajectories/ (all its agents), a directory, or an exported flat .csv / .csv.gz file; repeat it to combine.

Why not PPO on old data: PPO needs the action probabilities of the policy that produced the steps; replaying a file would silently train on
stale samples. Cloning only needs (observation -> action) pairs, so it works on any data. The value warm-up fits the critic to the
discounted returns of the logged rewards so a later PPO run does not start from a random critic.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from stable_baselines3 import PPO

from src.training import data as D
from src.training.weights_io import resolve_checkpoint
from src.utils import config as C


def finetune(checkpoint, sources, out_name: str, *, bc: bool = True, value: bool = True, epochs: int = 30, lr: float = 1e-3, batch: int = 256,
             device: str = "cpu", seed: int = 0, log=print, log_std: float | None = None, low_weight: float = 1.0, low_height_m: float = 400.0, **filters) -> dict:
    src = resolve_checkpoint(checkpoint)
    model = PPO.load(str(src), device=device)
    pol = model.policy
    ds = D.load_dataset(sources, gamma=model.gamma, **filters)
    n = len(ds["obs"])
    if n == 0:
        raise ValueError("no steps matched: check the data source and the filters")
    log(f"[data] {ds['episodes']} episodes, {n} steps (filters: {filters or 'none'})")
    want = int(model.observation_space.shape[0])
    if ds["obs"].shape[1] != want or ds["act"].shape[1] != int(model.action_space.shape[0]):
        raise ValueError(f"the data has {ds['obs'].shape[1]} observation and {ds['act'].shape[1]} action columns but {src.stem} expects {want} and {model.action_space.shape[0]} "
                         f"({'Dual land' if want == C.DUAL_OBS_DIM else 'single lander'} checkpoint vs {'Dual land' if ds['obs'].shape[1] == C.DUAL_OBS_DIM else 'single lander'} data): "
                         "clone a dual checkpoint from dual data, and a single one from single data (python -m src.sim.dual_init converts a single checkpoint to a dual one)")
    obs = torch.as_tensor(ds["obs"], device=device)
    act = torch.as_tensor(np.clip(ds["act"], -1.0, 1.0), device=device)
    ret = torch.as_tensor(ds["ret"], device=device)
    ret_scale = float(ret.abs().mean().clamp(min=1e-3))
    hcols = [1] + ([19] if obs.shape[1] == C.DUAL_OBS_DIM else [])                                  # the height of every lander (Dual land: A at 1, B at 19)
    low = torch.stack([obs[:, c] * C.OBS_Y_TO_M < low_height_m for c in hcols]).any(dim=0)
    wts = torch.where(low, torch.tensor(float(low_weight)), torch.tensor(1.0)).to(device)           # the last metres decide a landing: weight them more
    wts = wts / wts.mean()
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)

    def losses(idx):
        o = obs[idx]
        out = {}
        if bc:
            mean = pol.get_distribution(o).distribution.mean
            out["bc"] = (((mean - act[idx]) ** 2).mean(dim=1) * wts[idx]).mean()
        if value:
            out["value"] = torch.nn.functional.mse_loss(pol.predict_values(o).squeeze(-1) / ret_scale, ret[idx] / ret_scale)
        return out

    def evaluate():
        with torch.no_grad():
            res = {k: float(v) for k, v in losses(torch.arange(n, device=device)).items()}
        return res

    pol.set_training_mode(True)
    before = evaluate()
    log(f"[before] {json.dumps({k: round(v, 5) for k, v in before.items()})}")
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.as_tensor(rng.permutation(n), device=device)
        for i in range(0, n, batch):
            ls = losses(perm[i:i + batch])
            loss = sum(ls.values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            log(f"[epoch {ep + 1}/{epochs}] {json.dumps({k: round(v, 5) for k, v in evaluate().items()})}")
    after = evaluate()
    pol.set_training_mode(False)
    if log_std is not None:                  # cloning leaves the exploration noise at its initial size; shrink it so PPO starts near the cloned behaviour
        with torch.no_grad():
            pol.log_std.fill_(log_std)
        log(f"[noise] exploration std set to {float(np.exp(log_std)):.2f}")
    C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(C.CHECKPOINT_DIR / out_name))
    C.MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    report = dict(source=src.name, out=out_name + ".zip", sources=[str(s) for s in sources], filters=filters, steps=n, episodes=ds["episodes"],
                  bc=bc, value=value, epochs=epochs, lr=lr, before=before, after=after, seconds=round(time.time() - t0, 1))
    (C.MODEL_CONFIG_DIR / f"{out_name}.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"[done] saved {out_name}.zip in {report['seconds']} s")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--no-bc", action="store_true")
    ap.add_argument("--no-value", action="store_true")
    ap.add_argument("--min-merit", type=float, default=None)
    ap.add_argument("--soft-only", action="store_true")
    ap.add_argument("--success-only", action="store_true")
    ap.add_argument("--min-return", type=float, default=None)
    ap.add_argument("--max-return", type=float, default=None)
    ap.add_argument("--kept", default=None, help="only episodes kept for these reasons, e.g. best or best+sample")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--low-weight", type=float, default=1.0, help="weight of samples below 400 m in the cloning loss")
    ap.add_argument("--log-std", type=float, default=None, help="set the policy's exploration log-std after cloning (e.g. -1.5 = std 0.22)")
    a = ap.parse_args()
    if a.no_bc and a.no_value:
        sys.exit("nothing to train: both --no-bc and --no-value given")
    flt = dict(min_merit=a.min_merit, soft_only=a.soft_only or None, success_only=a.success_only or None, min_return=a.min_return,
               max_return=a.max_return, kept_for=a.kept.split("+") if a.kept else None)
    flt = {k: v for k, v in flt.items() if v is not None}
    finetune(a.checkpoint, a.data, a.out, bc=not a.no_bc, value=not a.no_value, epochs=a.epochs, lr=a.lr, batch=a.batch, device=a.device,
             log=lambda s: print(s, flush=True), log_std=a.log_std, low_weight=a.low_weight, **flt)


if __name__ == "__main__":
    main()
