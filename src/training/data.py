"""Experience data: a sampled step-level trajectory logger, and readers / exporters for it.

Layout (one directory per run, or per agent of a sweep):
    data/trajectories/<run>/manifest.json       what produced the data (reward fingerprint, constants, columns)
    data/trajectories/<run>/episodes_kept.csv   one row per kept episode (outcome, return, why it was kept, which chunk)
    data/trajectories/<run>/traj_00001.csv.gz   step rows: ep, t, o0..o13 (what the policy saw), a0, a1 (what it did), r (scaled reward), done

Exported ("flat") CSV = the step columns plus the episode's outcome repeated on every row, so a single .csv file can be loaded
back by the cloning tools with the same filters.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import time
from collections import deque
from pathlib import Path

import numpy as np

from src.utils import config as C

N_OBS, N_ACT = {"dual": (C.DUAL_OBS_DIM, C.DUAL_ACT_DIM), "pair": (C.DUAL_OBS_DIM, 5)}.get(C.START_PROFILE, (18, 5))
OBS_COLS = [f"o{i}" for i in range(N_OBS)]
ACT_COLS = [f"a{i}" for i in range(N_ACT)]
STEP_COLS = ["ep", "t", *OBS_COLS, *ACT_COLS, "r", "done"]
META_COLS = ["merit", "soft", "success", "end", "ret", "level", "rules_stage", "kept_for"]
EPISODE_COLS = ["ep", "timesteps", "level", "rules_stage", "end", "merit", "soft", "success", "ret", "steps", "kept_for", "file"]
ROOT_DIR = C.DATA_DIR / "trajectories"
_COL_RE = {"o": re.compile(r"^o(\d+)$"), "a": re.compile(r"^a(\d+)$")}


def num_cols(columns, prefix: str) -> list:
    """The o0..oN (observation) or a0..aN (action) columns of a file, in order: a trajectory file carries whatever dimensions its run had
    (18 / 5 for one lander, 40 / 10 for Dual land), so readers detect them instead of assuming."""
    found = sorted((int(m.group(1)), c) for c in columns if (m := _COL_RE[prefix].match(str(c))))
    return [c for _, c in found]


def kind_of(n_obs: int, n_act: int = 5) -> str:
    """single = one lander (18 -> 5); dual = ONE network for both Dual land landers (40 -> 10); agent = one lander's network in a two-network pair (40 -> 5)"""
    return "single" if (n_obs, n_act) == (18, 5) else "dual" if (n_obs, n_act) == (C.DUAL_OBS_DIM, C.DUAL_ACT_DIM) else "agent" if (n_obs, n_act) == (C.DUAL_OBS_DIM, 5) else "other"


def reward_constants() -> dict:
    """The numbers of the custom reward (for the manifest, so data can be matched to the rules that produced it)."""
    from src.training import reward as R
    return {k: v for k, v in vars(R).items() if k.isupper() and isinstance(v, (int, float, list, tuple))}


def fingerprint(constants: dict | None = None) -> str:
    blob = json.dumps(dict(reward=constants if constants is not None else reward_constants(), overrides=C.OVERRIDES), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def dir_for(run_dir: Path) -> Path:
    """data/trajectories/<same relative path as the log dir>"""
    run_dir = Path(run_dir)
    try:
        rel = run_dir.resolve().relative_to(C.LOG_DIR.resolve())
    except ValueError:
        rel = Path(run_dir.name)
    return ROOT_DIR / rel


def total_bytes(root: Path | None = None) -> int:
    root = ROOT_DIR if root is None else root
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) if root.exists() else 0


def enforce_budget(budget_bytes: float, root: Path | None = None, keep: Path | None = None) -> list:
    """Delete the oldest chunk files (by modification time) until everything under `root` fits. Never touches `keep`."""
    root = ROOT_DIR if root is None else root
    if not root.exists():
        return []
    chunks = sorted((f for f in root.rglob("traj_*.csv.gz") if f != keep), key=lambda f: f.stat().st_mtime)
    used, removed = total_bytes(root), []
    for f in chunks:
        if used <= budget_bytes:
            break
        used -= f.stat().st_size
        f.unlink(missing_ok=True)
        removed.append(f)
    return removed


class TrajectoryLogger:
    """Collects every episode's steps in memory, decides at the end of the episode whether to keep it, writes kept ones in chunks."""

    def __init__(self, run_dir: Path, name: str, every: int | None = None, tail: float | None = None, budget_gb: float | None = None,
                 flush_episodes: int = 40, flush_seconds: float = 60.0, window: int = 500):
        self.dir = dir_for(run_dir)
        self.name = name
        self.every = max(1, int(C.DATA_EVERY if every is None else every))
        self.tail = C.DATA_TAIL if tail is None else tail
        self.budget = (C.DATA_BUDGET_GB if budget_gb is None else budget_gb) * 1e9
        self.flush_episodes, self.flush_seconds = flush_episodes, flush_seconds
        self.window = deque(maxlen=window)
        self.buf: dict = {}
        self.pending: list = []
        self.n_seen = 0
        self.n_kept = 0
        self.last_flush = time.time()
        self.lo = self.hi = None
        self.dir.mkdir(parents=True, exist_ok=True)
        existing = [int(f.name[5:10]) for f in self.dir.glob("traj_*.csv.gz") if f.name[5:10].isdigit()]
        self.chunk = max(existing, default=0)
        eps = self.dir / "episodes_kept.csv"
        self.ep_no = 0
        if eps.exists():
            try:
                with open(eps, newline="") as f:
                    self.ep_no = max((int(r["ep"]) for r in csv.DictReader(f)), default=0)
            except (OSError, ValueError, KeyError):
                pass
        self._manifest()

    def _manifest(self):
        consts = reward_constants()
        (self.dir / "manifest.json").write_text(json.dumps(dict(
            run=self.name, updated=time.strftime("%Y-%m-%d %H:%M:%S"), reward_fingerprint=fingerprint(consts), reward_constants=consts,
            overrides=C.OVERRIDES, columns=getattr(self, "step_cols", STEP_COLS), episode_columns=EPISODE_COLS, sampling=dict(every=self.every, tail=self.tail),
            gamma=C.PPO_KWARGS.get("gamma"), seen=self.n_seen, kept=self.n_kept), indent=1, default=str))

    # ---- recording
    def step(self, env_idx: int, obs, act, reward: float) -> None:
        self.buf.setdefault(env_idx, []).append((np.asarray(obs, dtype=np.float32).copy(), np.asarray(act, dtype=np.float32).copy(), float(reward)))

    def end_episode(self, env_idx: int, metrics: dict, num_timesteps: int) -> None:
        steps = self.buf.pop(env_idx, [])
        if not steps:
            return
        ret = float(sum(s[2] for s in steps))
        self.n_seen += 1
        why = []
        if self.n_seen % self.every == 0:
            why.append("sample")
        self.window.append(ret)
        if len(self.window) >= 100 and (self.n_seen % 20 == 0 or self.lo is None):
            self.lo, self.hi = (float(x) for x in np.quantile(self.window, [self.tail, 1.0 - self.tail]))
        if self.lo is not None:
            if ret <= self.lo:
                why.append("worst")
            elif ret >= self.hi:
                why.append("best")
        if not why:
            return
        self.ep_no += 1
        self.n_kept += 1
        self.pending.append(dict(ep=self.ep_no, steps=steps, ret=ret, why="+".join(why), timesteps=int(num_timesteps), m=metrics))
        if len(self.pending) >= self.flush_episodes or time.time() - self.last_flush > self.flush_seconds:
            self.flush()

    # ---- writing
    def flush(self) -> None:
        if not self.pending:
            return
        self.chunk += 1
        path = self.dir / f"traj_{self.chunk:05d}.csv.gz"
        rows = []
        n_obs, n_act = len(self.pending[0]["steps"][0][0]), len(self.pending[0]["steps"][0][1])
        self.step_cols = ["ep", "t", *[f"o{i}" for i in range(n_obs)], *[f"a{i}" for i in range(n_act)], "r", "done"]
        with gzip.open(path, "wt", newline="", compresslevel=6) as f:
            w = csv.writer(f)
            w.writerow(self.step_cols)
            for e in self.pending:
                last = len(e["steps"]) - 1
                for t, (o, a, r) in enumerate(e["steps"]):
                    w.writerow([e["ep"], t, *(f"{v:.6g}" for v in o), *(f"{v:.5g}" for v in a), f"{r:.6g}", int(t == last)])
                m = e["m"]
                rows.append([e["ep"], e["timesteps"], m.get("level"), m.get("rules_stage"), m.get("end"), f"{float(m.get('merit', 0.0)):.3f}",
                             int(bool(m.get("soft"))), int(bool(m.get("success"))), f"{e['ret']:.4f}", len(e["steps"]), e["why"], path.name])
        eps = self.dir / "episodes_kept.csv"
        new = not eps.exists()
        with open(eps, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(EPISODE_COLS)
            w.writerows(rows)
        self.pending = []
        self.last_flush = time.time()
        enforce_budget(self.budget, keep=path)
        self._manifest()


# ---------------------------------------------------------------- reading
def source_dirs(src) -> list:
    """A run's data directory, or a sweep run's directory containing agent_<i> data directories."""
    p = Path(src)
    if not p.is_absolute() and not p.exists():
        p = ROOT_DIR / src
    if p.is_dir():
        kids = sorted(d for d in p.iterdir() if d.is_dir() and (d / "episodes_kept.csv").exists())
        return kids if kids else [p]
    return []


def _episode_table(d: Path) -> dict:
    out = {}
    try:
        with open(d / "episodes_kept.csv", newline="") as f:
            for r in csv.DictReader(f):
                out[int(r["ep"])] = r
    except (OSError, KeyError, ValueError):
        pass
    return out


def _passes(meta: dict, f: dict) -> bool:
    try:
        if f.get("min_merit") is not None and float(meta.get("merit", 0.0)) < f["min_merit"]:
            return False
        if f.get("soft_only") and not int(float(meta.get("soft", 0))):
            return False
        if f.get("success_only") and not int(float(meta.get("success", 0))):
            return False
        ret = float(meta.get("ret", 0.0))
        if f.get("min_return") is not None and ret < f["min_return"]:
            return False
        if f.get("max_return") is not None and ret > f["max_return"]:
            return False
        if f.get("kept_for") and not (set(str(meta.get("kept_for", "")).split("+")) & set(f["kept_for"])):
            return False
        if f.get("ends") and str(meta.get("end", "")) not in f["ends"]:
            return False
    except (TypeError, ValueError):
        return False
    return True


def _episode_arrays(g, ocols, acols) -> dict:
    return dict(obs=g[ocols].to_numpy(np.float32), act=g[acols].to_numpy(np.float32), rew=g["r"].to_numpy(np.float32))


def iter_episodes(sources, **filters):
    """Yields dict(obs (T, n_obs), act (T, n_act), rew (T,), meta). `sources`: run name / directory / flat .csv(.gz) file, or a list of them.
    The observation and action sizes come from each file's own columns."""
    import pandas as pd
    for src in (sources if isinstance(sources, (list, tuple)) else [sources]):
        p = Path(src)
        if p.is_file():
            df = pd.read_csv(p)
            ocols, acols = num_cols(df.columns, "o"), num_cols(df.columns, "a")
            missing = [c for c in ("ep", "t", "r", "done") if c not in df.columns] + ([] if ocols else ["o0"]) + ([] if acols else ["a0"])
            if missing:
                raise ValueError(f"{p.name}: missing columns {missing}")
            for ep, g in df.groupby("ep", sort=False):
                meta = {c: g[c].iloc[0] for c in META_COLS if c in g.columns}
                if _passes(meta, filters):
                    yield dict(**_episode_arrays(g, ocols, acols), meta=meta)
            continue
        for d in source_dirs(src):
            table = _episode_table(d)
            for chunk in sorted(d.glob("traj_*.csv.gz")):
                df = pd.read_csv(chunk)
                ocols, acols = num_cols(df.columns, "o"), num_cols(df.columns, "a")
                for ep, g in df.groupby("ep", sort=False):
                    meta = table.get(int(ep), {})
                    if _passes(meta, filters):
                        yield dict(**_episode_arrays(g, ocols, acols), meta=meta)


def discounted_returns(rew: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(rew, dtype=np.float32)
    acc = 0.0
    for i in range(len(rew) - 1, -1, -1):
        acc = rew[i] + gamma * acc
        out[i] = acc
    return out


def load_dataset(sources, gamma: float = 0.999, **filters) -> dict:
    """Stacks the filtered episodes: obs (N, n_obs), act (N, n_act), ret (N,) discounted returns of the logged (scaled) rewards.
    All sources must have the same observation / action sizes (a single-lander and a Dual land file cannot be mixed)."""
    obs, act, ret, n_eps = [], [], [], 0
    for e in iter_episodes(sources, **filters):
        if obs and (e["obs"].shape[1] != obs[0].shape[1] or e["act"].shape[1] != act[0].shape[1]):
            raise ValueError(f"the data mixes {obs[0].shape[1]}-input and {e['obs'].shape[1]}-input files (single lander and Dual land): clone from one kind at a time")
        obs.append(e["obs"]); act.append(e["act"]); ret.append(discounted_returns(e["rew"], gamma)); n_eps += 1
    if not obs:
        return dict(obs=np.zeros((0, N_OBS), np.float32), act=np.zeros((0, N_ACT), np.float32), ret=np.zeros(0, np.float32), episodes=0)
    return dict(obs=np.concatenate(obs), act=np.concatenate(act), ret=np.concatenate(ret), episodes=n_eps)


def dataset_info(src) -> dict:
    """What a data source holds: observation / action sizes and the kind ('single', 'dual'), read from the file header or the run's first chunk."""
    import pandas as pd
    p = Path(src)
    if not p.is_absolute() and not p.exists():
        p = ROOT_DIR / src
    cols = None
    try:
        if p.is_file():
            cols = list(pd.read_csv(p, nrows=0).columns)
        else:
            for d in source_dirs(p):
                try:
                    cols = list(json.loads((d / "manifest.json").read_text()).get("columns") or [])
                except (OSError, ValueError):
                    cols = None
                if not num_cols(cols or [], "o"):
                    first = next(iter(sorted(d.glob("traj_*.csv.gz"))), None)
                    cols = list(pd.read_csv(first, nrows=0).columns) if first else None
                if cols:
                    break
    except (OSError, ValueError):
        cols = None
    n_obs, n_act = len(num_cols(cols or [], "o")), len(num_cols(cols or [], "a"))
    return dict(n_obs=n_obs, n_act=n_act, kind=kind_of(n_obs, n_act) if n_obs else "unknown")


def export_flat_csv(sources, out, **filters) -> dict:
    """Writes the flat CSV (step columns + the episode outcome on every row) to a path or an open text stream. Returns counts."""
    close = False
    if isinstance(out, (str, Path)):
        out, close = open(out, "w", newline=""), True
    try:
        w = csv.writer(out)
        eps = steps = 0
        wrote_header = False
        for e in iter_episodes(sources, **filters):
            if not wrote_header:                                              # the header follows the data's own sizes
                w.writerow(["ep", "t", *[f"o{i}" for i in range(e["obs"].shape[1])], *[f"a{i}" for i in range(e["act"].shape[1])], "r", "done", *META_COLS])
                wrote_header = True
            eps += 1
            m = e["meta"]
            tail = [m.get(c, "") for c in META_COLS]
            T = len(e["rew"])
            for t in range(T):
                w.writerow([eps, t, *(f"{v:.6g}" for v in e["obs"][t]), *(f"{v:.5g}" for v in e["act"][t]), f"{e['rew'][t]:.6g}", int(t == T - 1), *tail])
            steps += T
        if not wrote_header:
            w.writerow(STEP_COLS + META_COLS)
        return dict(episodes=eps, steps=steps)
    finally:
        if close:
            out.close()


def run_summaries() -> list:
    """What the Data tab lists: every run that has trajectory data."""
    out = []
    if not ROOT_DIR.exists():
        return out
    for d in sorted(ROOT_DIR.iterdir()):
        if not d.is_dir():
            continue
        parts = source_dirs(d)
        files = [f for q in parts for f in q.glob("traj_*.csv.gz")]
        if not files:
            continue
        eps = sum(len(_episode_table(q)) for q in parts)
        man = {}
        try:
            man = json.loads((parts[0] / "manifest.json").read_text())
        except (OSError, ValueError):
            pass
        info = dataset_info(d)
        out.append(dict(run=d.name, agents=len(parts) if len(parts) > 1 or parts[0] != d else 0, episodes=eps, files=len(files), kind=info["kind"], n_obs=info["n_obs"], n_act=info["n_act"],
                        bytes=sum(f.stat().st_size for f in files), fingerprint=man.get("reward_fingerprint"), updated=max(f.stat().st_mtime for f in files)))
    return out
