"""Background jobs behind the app's Pipeline and Checkpoints tabs (kept apart from server.py).

Pipeline: runs `python -m src.sim.pipeline` (expert demonstrations -> cloning -> DAgger -> evaluation) and reads its status.json.
Checkpoints: list with kind / size / exploration noise / last evaluation, make a Dual land copy of a single-lander checkpoint, make a quiet
(low exploration noise) copy, evaluate in the background, move to the trash folder. Nothing here overwrites an existing checkpoint.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from src.sim import pipeline as PL
from src.training import weights_io as WIO
from src.utils import config as C

ROOT = Path(__file__).resolve().parents[2]
NAME_RE = re.compile(r"[A-Za-z0-9_\-]{1,60}")
PIPE_DIR = C.LOG_DIR / "pipeline"
TRASH_DIR = C.MODEL_CONFIG_DIR.parent / "_trash"
PIPE = dict(proc=None, name=None)
EVAL = dict(proc=None, name=None, started=None, out=None)


def _popen(argv, out_path: Path, env_extra: dict | None = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore", **(env_extra or {}))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    return subprocess.Popen(argv, cwd=ROOT, env=env, stdout=open(out_path, "w"), stderr=subprocess.STDOUT, creationflags=flags)


def _kill_tree(p) -> None:
    try:
        proc = psutil.Process(p.pid)
        kids = proc.children(recursive=True)
    except psutil.Error:
        return
    for q in kids + [proc]:
        try:
            q.kill()
        except psutil.Error:
            pass


def _running(slot: dict) -> bool:
    return slot["proc"] is not None and slot["proc"].poll() is None


def _num(src: dict, key: str, lo: float, hi: float, default, as_int: bool = False):
    v = src.get(key, default)
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{key}: not a number")
    if not lo <= x <= hi:
        raise ValueError(f"{key} must be between {lo:g} and {hi:g}")
    return int(x) if as_int else x


def _levels(s, what: str) -> str:
    s = str(s).replace(" ", "")
    if not re.fullmatch(r"[0-9](,[0-9])*", s) or any(int(x) >= C.N_LEVELS for x in s.split(",")):
        raise ValueError(f"{what}: comma-separated levels 0-{C.N_LEVELS - 1}, for example 0,0,1")
    return s


def _kind(name: str) -> str:
    from src.training.pair import resolve_pair
    if not (C.CHECKPOINT_DIR / f"{name}.zip").exists() and resolve_pair(name):
        return "pair"
    return WIO.checkpoint_info(name)["kind"]


# ------------------------------------------------------------------ pipeline
def pipeline_start(body: dict, env_extra: dict | None = None) -> tuple:
    if _running(PIPE):
        return 409, dict(errors=[f"The pipeline {PIPE['name']} is still running. Stop it first."])
    name = str(body.get("name", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-]{2,40}", name):
        raise ValueError("name: 2-40 letters, digits, _ and -")
    pair = bool(body.get("pair", False))
    dual = bool(body.get("dual", False)) and not pair
    steps = [s for s in PL.STEP_ORDER if s in (body.get("steps") or PL.STEP_ORDER)]
    if not steps:
        raise ValueError("tick at least one step")
    cfg = PL.merged(dict(name=name, dual=dual or pair, pair=pair, steps=steps, seed=_num(body, "seed", 0, 10_000, 0, True), overwrite=bool(body.get("overwrite", False))))
    e, c, d, ev = (body.get(k) or {} for k in ("expert", "clone", "dagger", "eval"))
    cfg["expert"] = dict(episodes=_num(e, "episodes", 10, 5000, cfg["expert"]["episodes"], True), noise=_num(e, "noise", 0, 1, cfg["expert"]["noise"]),
                         levels=_levels(e.get("levels", cfg["expert"]["levels"]), "expert levels"))
    cfg["clone"] = dict(epochs=_num(c, "epochs", 1, 500, cfg["clone"]["epochs"], True), lr=_num(c, "lr", 1e-6, 0.1, cfg["clone"]["lr"]),
                        batch=_num(c, "batch", 32, 8192, cfg["clone"]["batch"], True), log_std=_num(c, "log_std", -4, 0, cfg["clone"]["log_std"]),
                        low_weight=_num(c, "low_weight", 1, 50, cfg["clone"]["low_weight"]))
    cfg["dagger"] = dict(rounds=_num(d, "rounds", 1, 20, cfg["dagger"]["rounds"], True), episodes=_num(d, "episodes", 10, 2000, cfg["dagger"]["episodes"], True),
                         epochs=_num(d, "epochs", 1, 200, cfg["dagger"]["epochs"], True), lr=_num(d, "lr", 1e-6, 0.1, cfg["dagger"]["lr"]),
                         batch=_num(d, "batch", 32, 8192, cfg["dagger"]["batch"], True), noise=_num(d, "noise", 0, 1, cfg["dagger"]["noise"]),
                         levels=_levels(d.get("levels", cfg["dagger"]["levels"]), "DAgger levels"), log_std=_num(d, "log_std", -4, 0, cfg["dagger"]["log_std"]),
                         low_weight=_num(d, "low_weight", 1, 50, cfg["dagger"]["low_weight"]))
    cfg["eval"] = dict(episodes=_num(ev, "episodes", 5, 500, cfg["eval"]["episodes"], True), levels=_levels(ev.get("levels", cfg["eval"]["levels"]), "evaluation levels"))
    init, single = str(body.get("init", "") or ""), str(body.get("from_single", "") or "")
    for label, ck in (("start checkpoint", init), ("single-lander checkpoint", single)):
        if ck and (not NAME_RE.fullmatch(ck) or not _exists(ck)):
            raise ValueError(f"no checkpoint named {ck} ({label})")
    want = "pair" if pair else "dual" if dual else "single"
    if init and _kind(init) != want:
        raise ValueError(f"{init} is a {_kind(init)} checkpoint; this pipeline is {want}. " + ("Make a dual one or a pair in the Checkpoints tab first." if dual or pair else ""))
    if single and not (dual or pair):
        raise ValueError("a single-lander checkpoint to start from is only used by dual and pair pipelines")
    if single and _kind(single) != "single":
        raise ValueError(f"{single} is not a single-lander checkpoint")
    cfg["init"], cfg["from_single"] = init, single
    if "expert" not in steps and not any((C.DATA_DIR / "trajectories" / f"{name}_expert{s}").is_dir() for s in (("_A", "_B") if pair else ("",))):
        raise ValueError(f"the expert step is off but there are no demonstrations named {name}_expert yet")
    taken = PL.outputs_of(name)
    if not cfg["overwrite"] and "expert" in steps and (taken["checkpoints"] or taken["data"]):
        names = [f.stem for f in taken["checkpoints"]] + [d.name for d in taken["data"]]
        return 409, dict(errors=[f"{name} already has outputs ({', '.join(names[:6])}{'...' if len(names) > 6 else ''}). Pick another name or tick 'replace the old outputs'."])
    d = PIPE_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    (d / "status.json").unlink(missing_ok=True)
    PIPE.update(proc=_popen([sys.executable, "-m", "src.sim.pipeline", "--config", str(d / "config.json")], PIPE_DIR / f"{name}.out", env_extra), name=name)
    return 200, dict(name=name, steps=steps, dual=dual, pair=pair)


def _read_status(name: str) -> dict | None:
    try:
        return json.loads((PIPE_DIR / name / "status.json").read_text())
    except (OSError, ValueError):
        return None


def pipeline_status(name: str | None = None) -> dict:
    name = name or PIPE["name"]
    if not name:
        names = pipeline_names()
        name = names[0] if names else None
    if not name:
        return dict(state="none")
    st = _read_status(name)
    alive = _running(PIPE) and PIPE["name"] == name
    if st is None:
        out = PIPE_DIR / f"{name}.out"
        tail = out.read_text(errors="ignore").splitlines()[-8:] if out.exists() else []
        return dict(state="starting" if alive else "failed", name=name, log=tail, alive=alive)
    if st["state"] == "running" and not alive:
        st["state"] = "stopped"                                  # the process is gone without having finished: killed or crashed
    st["alive"] = alive
    return st


def pipeline_stop() -> dict:
    if not _running(PIPE):
        return dict(stopped=False)
    _kill_tree(PIPE["proc"])
    return dict(stopped=True, name=PIPE["name"])


def pipeline_names() -> list:
    if not PIPE_DIR.exists():
        return []
    ds = [d for d in PIPE_DIR.iterdir() if d.is_dir() and (d / "status.json").exists()]
    return [d.name for d in sorted(ds, key=lambda d: -(d / "status.json").stat().st_mtime)]


def pipeline_list() -> list:
    out = []
    for n in pipeline_names():
        st = _read_status(n) or {}
        ev = st.get("final_eval") or {}
        out.append(dict(name=n, state=st.get("state"), dual=st.get("dual"), pair=bool((st.get("config") or {}).get("pair")), started=st.get("started"), final=st.get("final"), score=ev.get("score"),
                        rounds=len(st.get("rounds", []))))
    return out


# ------------------------------------------------------------------ checkpoints
def _eval_summary(name: str) -> dict | None:
    f = C.MODEL_CONFIG_DIR / f"{name}.eval.json"
    try:
        r = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    return dict(score=r.get("score"), episodes=r.get("episodes"), when=f.stat().st_mtime,
                levels=[dict(level=x["level"], success=x["success"], n=x["n"], speed=x.get("speed"), gap=x.get("gap_median")) for x in r.get("levels", [])])


def _origin(name: str) -> str:
    f = C.MODEL_CONFIG_DIR / f"{name}.json"
    try:
        r = json.loads(f.read_text())
    except (OSError, ValueError):
        return ""
    if "source" in r and "epochs" in r:
        return f"cloned from {str(r['source']).replace('.zip', '')} ({r.get('episodes')} episodes, {r.get('epochs')} epochs)"
    if "reward" in r:
        return f"PPO run, {r.get('total_steps', '?')} steps requested" + (f", started from {r['init_from']}" if r.get("init_from") else "")
    return ""


def checkpoint_details(training_run: str | None = None) -> list:
    if not C.CHECKPOINT_DIR.exists():
        return []
    infos = {}
    for f in sorted(C.CHECKPOINT_DIR.glob("*.zip"), key=lambda f: -f.stat().st_mtime):
        try:
            info = WIO.checkpoint_info(f.stem)
        except Exception as e:                                   # a damaged file must not hide the others
            info = dict(name=f.stem, bytes=f.stat().st_size, mtime=f.stat().st_mtime, backup=".bak_" in f.name, kind="unreadable", n_obs=0, n_act=0, arch=[], std=None, error=str(e)[:100])
        infos[f.stem] = info
    rows, used = [], set()
    for name, info in infos.items():                             # two networks of a pair (<name>_A and <name>_B) are shown as ONE row
        if name.endswith("_A") and info["kind"] == "agent" and f"{name[:-2]}_B" in infos and infos[f"{name[:-2]}_B"]["kind"] == "agent":
            b = infos[f"{name[:-2]}_B"]
            base = name[:-2]
            used.update((name, b["name"]))
            rows.append(dict(info, name=base, kind="pair", bytes=info["bytes"] + b["bytes"], mtime=max(info["mtime"], b["mtime"]), std=round((info["std"] + b["std"]) / 2, 4),
                             files=[name, b["name"]], arch=info["arch"]))
    rows += [i for n, i in infos.items() if n not in used]
    rows.sort(key=lambda r: -r["mtime"])
    for info in rows:
        info.update(origin=_origin(info["name"]), eval=_eval_summary(info["name"]), busy=bool(training_run and info["name"].startswith(training_run)))
    return rows


def _check_name(name: str, what: str = "name") -> str:
    if not NAME_RE.fullmatch(name or ""):
        raise ValueError(f"{what}: letters, digits, _ and - only (max 60)")
    return name


def _exists(name: str) -> bool:
    return (C.CHECKPOINT_DIR / f"{name}.zip").exists() or all((C.CHECKPOINT_DIR / f"{name}_{s}.zip").exists() for s in "AB")


def _is_pair(name: str) -> bool:
    return not (C.CHECKPOINT_DIR / f"{name}.zip").exists() and all((C.CHECKPOINT_DIR / f"{name}_{s}.zip").exists() for s in "AB")


def ckpt_make_pair(body: dict) -> dict:
    from src.training.pair import make_pair
    src, out = _check_name(str(body.get("from", "")), "source"), _check_name(str(body.get("out", "")), "new name")
    if not (C.CHECKPOINT_DIR / f"{src}.zip").exists():
        raise ValueError(f"no checkpoint named {src}")
    return make_pair(src, out)


def ckpt_make_dual(body: dict) -> dict:
    from src.sim.dual_init import make_dual
    src, out = _check_name(str(body.get("from", "")), "source"), _check_name(str(body.get("out", "")), "new name")
    if not (C.CHECKPOINT_DIR / f"{src}.zip").exists():
        raise ValueError(f"no checkpoint named {src}")
    return make_dual(src, out)


def ckpt_quiet(body: dict) -> dict:
    src, out = _check_name(str(body.get("from", "")), "source"), _check_name(str(body.get("out", "")), "new name")
    if not _exists(src):
        raise ValueError(f"no checkpoint named {src}")
    try:
        std = float(body.get("std", 0.08))
    except (TypeError, ValueError):
        raise ValueError("std: not a number")
    if _is_pair(src):                                            # both networks of a pair
        if _exists(out):
            raise ValueError(f"{out} already exists; pick another name")
        for s in "AB":
            WIO.save_with_std(f"{src}_{s}", f"{out}_{s}", std)
        return dict(checkpoint=out, std=std)
    return WIO.save_with_std(src, out, std)


def ckpt_delete(body: dict, training_run: str | None = None) -> dict:
    """Moves the checkpoint (and its settings / evaluation files) to models/_trash/<name>_<time>/: nothing is permanently deleted."""
    name = _check_name(str(body.get("name", "")))
    if not _exists(name):
        raise ValueError(f"no checkpoint named {name}")
    if training_run and name.startswith(training_run):
        raise ValueError(f"{name} belongs to the run that is training right now; stop it first")
    dest = TRASH_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    dest.mkdir(parents=True, exist_ok=True)
    files = [name] if not _is_pair(name) else [f"{name}_A", f"{name}_B"]
    for f in files:
        shutil.move(str(C.CHECKPOINT_DIR / f"{f}.zip"), dest / f"{f}.zip")
    for extra in (C.MODEL_CONFIG_DIR / f"{name}.json", C.MODEL_CONFIG_DIR / f"{name}.eval.json"):
        if extra.exists():
            shutil.move(str(extra), dest / extra.name)
    return dict(moved_to=str(dest.relative_to(ROOT)) if dest.is_relative_to(ROOT) else str(dest))


def ckpt_eval_start(body: dict) -> tuple:
    if _running(EVAL):
        return 409, dict(errors=[f"Already evaluating {EVAL['name']}."])
    name = _check_name(str(body.get("name", "")))
    if not _exists(name):
        raise ValueError(f"no checkpoint named {name}")
    episodes = _num(body, "episodes", 5, 500, 30, True)
    levels = _levels(body.get("levels", "0,1,2"), "levels")
    out = C.MODEL_CONFIG_DIR / f"{name}.eval.json"
    C.MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    EVAL.update(proc=_popen([sys.executable, "-m", "src.sim.evaluate", name, "--episodes", str(episodes), "--levels", levels, "--json", str(out)],
                            C.LOG_DIR / f"evaluate_{name}.out"), name=name, started=time.time(), out=out)
    return 200, dict(name=name, episodes=episodes, levels=levels)


def ckpt_eval_status() -> dict:
    if EVAL["name"] is None:
        return dict(state="none")
    running = _running(EVAL)
    try:
        lines = [l for l in (C.LOG_DIR / f"evaluate_{EVAL['name']}.out").read_text(errors="ignore").splitlines() if l.strip() and "Warning" not in l]
    except OSError:
        lines = []
    res = dict(state="running" if running else "done", name=EVAL["name"], log=lines[-10:], elapsed=int(time.time() - EVAL["started"]),
               code=None if running else EVAL["proc"].returncode)
    if not running and EVAL["proc"].returncode == 0:
        res["eval"] = _eval_summary(EVAL["name"])
    return res


# ------------------------------------------------------------------ showcase recorder (the "Showcase Recorder" screen)
REPLAY_SCRIPT = ROOT / "tools" / "record_replays.py"
REC = dict(proc=None, started=None, args=None, stopped=False)
REC_OUT = C.LOG_DIR / "record_replays.out"
REC_PLAN_RE = re.compile(r"^plan: (\d+) landings")
REC_STEP_RE = re.compile(r"^\s*(\d+)/(\d+)\s+level")


def replay_record_start(body: dict) -> tuple:
    """Starts tools/record_replays.py in the background (the same command a person would type). One run at a time."""
    if _running(REC):
        return 409, dict(errors=["A recording is already running."])
    facts_only = bool(body.get("facts_only", False))
    argv = [sys.executable, str(REPLAY_SCRIPT)]
    if facts_only:
        argv.append("--facts-only")
    else:
        argv += ["--per-level", str(_num(body, "per_level", 1, 4, 2, True)), "--failures", str(_num(body, "failures", 0, 3, 3, True))]
    REC.update(proc=_popen(argv, REC_OUT), started=time.time(), args=argv[2:], stopped=False)
    return 200, dict(args=REC["args"])


def replay_record_status() -> dict:
    if REC["proc"] is None:
        return dict(state="idle")
    running = _running(REC)
    try:
        lines = [l.rstrip() for l in REC_OUT.read_text(errors="ignore").splitlines() if l.strip() and "Warning" not in l and "warnings.warn" not in l]
    except OSError:
        lines = []
    done = total = 0
    for l in lines:
        m = REC_PLAN_RE.match(l)
        if m:
            total = int(m.group(1))
        m = REC_STEP_RE.match(l)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
    code = None if running else REC["proc"].returncode
    state = "running" if running else "stopped" if REC["stopped"] else "done" if code == 0 else "error"
    return dict(state=state, args=REC["args"], log=lines[-14:], done=done, total=total, elapsed=int(time.time() - REC["started"]), code=code)


def replay_record_stop() -> dict:
    if _running(REC):
        REC["stopped"] = True
        _kill_tree(REC["proc"])
    return replay_record_status()
