"""A model's weights as a CSV, and back.

Format (long form, one parameter per row):   tensor,shape,index,value
    mlp_extractor.policy_net.0.weight,64x11,0,0.123456789
`index` is the row-major position inside the tensor; `value` has 9 significant digits, which is exact for float32.
Importing needs the same tensor names and shapes as the target model. The target checkpoint is backed up first.
"""
from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from src.utils import config as C


def resolve_checkpoint(name_or_path) -> Path:
    p = Path(name_or_path)
    if p.suffix != ".zip":
        p = p.with_name(p.name + ".zip")
    if not p.is_absolute() and not p.exists():
        p = C.CHECKPOINT_DIR / p.name
    if not p.exists():
        raise FileNotFoundError(f"no checkpoint {name_or_path}")
    return p


def _load(path: Path):
    from stable_baselines3 import PPO
    return PPO.load(str(path), device="cpu")


def export_weights_csv(checkpoint, out) -> dict:
    """Writes every tensor of the policy (actor, critic, log-std) to `out` (path or text stream)."""
    model = _load(resolve_checkpoint(checkpoint))
    close = False
    if isinstance(out, (str, Path)):
        out, close = open(out, "w", newline=""), True
    try:
        w = csv.writer(out)
        w.writerow(["tensor", "shape", "index", "value"])
        n = 0
        for name, t in model.policy.state_dict().items():
            a = t.detach().cpu().numpy().astype(np.float32).ravel()
            shape = "x".join(str(d) for d in t.shape) or "1"
            for i, v in enumerate(a):
                w.writerow([name, shape, i, f"{float(v):.9g}"])
            n += a.size
        return dict(tensors=len(model.policy.state_dict()), values=n)
    finally:
        if close:
            out.close()


def read_weights_csv(path) -> dict:
    """tensor name -> (shape tuple, flat float32 array)"""
    cols: dict = {}
    with open(path, newline="") as f:
        r = csv.reader(f)
        head = next(r, None)
        if head != ["tensor", "shape", "index", "value"]:
            raise ValueError("not a weights CSV (expected the header tensor,shape,index,value)")
        for row in r:
            name, shape, idx, val = row
            entry = cols.setdefault(name, [tuple(int(d) for d in shape.split("x")), {}])
            entry[1][int(idx)] = float(val)
    out = {}
    for name, (shape, vals) in cols.items():
        n = int(np.prod(shape))
        if sorted(vals) != list(range(n)):
            raise ValueError(f"{name}: expected {n} values for shape {'x'.join(map(str, shape))}, found {len(vals)} (missing or repeated indices)")
        out[name] = (shape, np.array([vals[i] for i in range(n)], dtype=np.float32))
    return out


def import_weights_csv(csv_path, checkpoint, out_name: str | None = None) -> dict:
    """Loads the CSV into the checkpoint. With out_name the result is a new checkpoint (nothing is overwritten); without it the
    checkpoint itself is replaced after a timestamped backup copy is made next to it."""
    src = resolve_checkpoint(checkpoint)
    new = read_weights_csv(csv_path)
    model = _load(src)
    sd = model.policy.state_dict()
    if set(new) != set(sd):
        raise ValueError(f"tensor names differ from the model's: missing {sorted(set(sd) - set(new))}, unexpected {sorted(set(new) - set(sd))}")
    for name, (shape, arr) in new.items():
        if tuple(sd[name].shape) != tuple(shape) and not (tuple(sd[name].shape) == () and shape == (1,)):
            raise ValueError(f"{name}: shape {shape} does not fit the model's {tuple(sd[name].shape)}")
    with torch.no_grad():
        for name, (shape, arr) in new.items():
            sd[name].copy_(torch.from_numpy(arr).reshape(sd[name].shape))
    backup = None
    if out_name:
        dest = C.CHECKPOINT_DIR / out_name
    else:
        backup = src.with_name(f"{src.stem}.bak_{time.strftime('%Y%m%d_%H%M%S')}.zip")
        shutil.copy2(src, backup)
        dest = src.with_suffix("")
    model.save(str(dest))
    return dict(checkpoint=str(dest.with_suffix(".zip").name), backup=backup.name if backup else None, tensors=len(new))


# ---------------------------------------------------------------- what a checkpoint is
_INFO_CACHE: dict = {}


def checkpoint_info(name_or_path) -> dict:
    """Name, kind ('single' = one lander, 'dual' = Dual land, 'other'), observation / action sizes, hidden layers, mean exploration std,
    file size and age. Cached per file and modification time (loading a model takes a moment)."""
    p = resolve_checkpoint(name_or_path)
    st = p.stat()
    key = (str(p), st.st_mtime_ns)
    if key not in _INFO_CACHE:
        model = _load(p)
        n_obs, n_act = int(model.observation_space.shape[0]), int(model.action_space.shape[0])
        arch = [int(m.out_features) for m in model.policy.mlp_extractor.policy_net if hasattr(m, "out_features")]
        std = float(torch.exp(model.policy.log_std.detach()).mean())
        kind = {(C.DUAL_OBS_DIM, C.DUAL_ACT_DIM): "dual", (18, 5): "single", (C.DUAL_OBS_DIM, 5): "agent"}.get((n_obs, n_act), "other")
        _INFO_CACHE[key] = dict(kind=kind, n_obs=n_obs, n_act=n_act, arch=arch, std=round(std, 4), timesteps=int(model.num_timesteps))
    return dict(name=p.stem, bytes=st.st_size, mtime=st.st_mtime, backup=".bak_" in p.name, **_INFO_CACHE[key])


def save_with_std(src, out_name: str, std: float) -> dict:
    """A copy of a checkpoint with its exploration noise set to `std` (PPO adds Gaussian noise of this size to every action; a cloned policy
    that lands cleanly without noise can fail with the default 0.2). Never overwrites an existing checkpoint."""
    src = resolve_checkpoint(src)
    dest = C.CHECKPOINT_DIR / f"{out_name}.zip"
    if dest.exists():
        raise ValueError(f"{out_name} already exists; pick another name")
    if not 0.005 <= std <= 1.0:
        raise ValueError("exploration std must be between 0.005 and 1.0")
    model = _load(src)
    with torch.no_grad():
        model.policy.log_std.fill_(float(np.log(std)))
    C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(dest.with_suffix("")))
    return dict(checkpoint=dest.name, std=std)
