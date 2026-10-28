"""SpaceXperiment control panel: start/stop training with tweakable parameters, run the tests and watch
the live view, all from the browser.

    python src/app/server.py --open          (or run_app.ps1)

Local only (127.0.0.1). POST endpoints require a JSON content type and a localhost Host header.
"""
import argparse
import csv
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import psutil

from src.app import camhulls, jobs, schema, startup
from src.training import data as D
from src.training import weights_io as WIO
from src.utils import config as C

HTML = Path(__file__).with_name("index.html")
STATE = {"run": None, "proc": None, "name": None, "started": None,
         "tests": {"state": "idle", "output": "", "started": None, "code": None}}
LOCK = threading.Lock()
PORT = 8767


# ------------------------------------------------------------------ helpers
def run_dirs() -> list[str]:
    if not C.LOG_DIR.exists():
        return []
    ds = [d for d in C.LOG_DIR.iterdir() if d.is_dir() and ((d / "live").exists() or (d / "episodes.csv").exists())]
    return [d.name for d in sorted(ds, key=lambda d: d.stat().st_mtime, reverse=True)]


def last_row(path: Path) -> dict | None:
    """Last complete row of a CSV, reading only the tail of the file."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 6000))
            lines = f.read().decode(errors="ignore").splitlines()
        head = open(path, newline="").readline().strip().split(",")
        for line in reversed(lines):
            row = next(csv.reader([line]))
            if len(row) == len(head) and row[0] != head[0]:
                return dict(zip(head, row))
    except (OSError, StopIteration):
        pass
    return None


def run_config(run: str) -> dict:
    try:
        return json.loads((C.MODEL_CONFIG_DIR / f"{run}.json").read_text())
    except (OSError, ValueError):
        return {}


def agent_dirs(run: str) -> list:
    d = C.LOG_DIR / run
    return sorted(d.glob("agent_*/"), key=lambda p: int(p.name.split("_")[1])) if d.exists() else []


def current_level(run: str) -> int:
    dirs = agent_dirs(run) or [C.LOG_DIR / run]
    lv = [int(r["level"]) for r in (last_row(d / "levels.csv") for d in dirs) if r]
    return max(lv) if lv else 0


def current_stage(run: str):
    """Rules stage of a run (None if it has no stages.csv yet). Sweeps: the highest stage among the agents."""
    dirs = agent_dirs(run) or [C.LOG_DIR / run]
    st = [int(r["stage"]) for r in (last_row(d / "stages.csv") for d in dirs) if r]
    return max(st) if st else None


_PROCS = {"t": 0.0, "v": []}
_PROCS_LOCK = threading.Lock()


def training_procs(fresh: bool = False) -> list:
    """Top-level train.py / sweep.py python processes (started here or elsewhere); pool workers and the
    Windows launcher/child pair are collapsed to their root. Scanning every process is slow with dozens
    of trainers running, so the result is cached for a few seconds unless fresh=True."""
    with _PROCS_LOCK:
        if not fresh and time.time() - _PROCS["t"] < 3.0:
            return [p for p in _PROCS["v"] if p.is_running()]
        cands = []
        for p in psutil.process_iter(["pid", "ppid", "name"]):
            if not (p.info["name"] or "").lower().startswith("python"):
                continue
            try:
                cmd = p.cmdline()
            except psutil.Error:
                continue
            if any(c.replace("\\", "/").endswith(("src/training/train.py", "src/training/sweep.py")) for c in cmd)                     and "multiprocessing" not in " ".join(cmd):
                p.info["cmdline"] = cmd
                cands.append(p)
        pids = {p.pid for p in cands}
        _PROCS.update(t=time.time(), v=[p for p in cands if p.info["ppid"] not in pids])
        return list(_PROCS["v"])


def kill_tree(p) -> None:
    try:
        kids = p.children(recursive=True)
    except psutil.Error:
        kids = []
    for q in kids + [p]:
        try:
            q.kill()
        except psutil.Error:
            pass


def run_arg(p, flag: str):
    cmd = p.info["cmdline"] or []
    return cmd[cmd.index(flag) + 1] if flag in cmd and cmd.index(flag) + 1 < len(cmd) else None


def train_status() -> dict:
    procs = training_procs()
    proc = STATE["proc"]
    ours = proc is not None and proc.poll() is None
    run = STATE["name"] if ours else None
    if not run and procs:
        sweep = any(str(c).endswith("sweep.py") for c in procs[0].info["cmdline"])
        run = f"{run_arg(procs[0], '--reward')}_s{run_arg(procs[0], '--seed')}{run_arg(procs[0], '--tag') or ''}" + ("_sweep" if sweep else "")
    running = bool(procs)
    st = dict(running=running, external=running and not ours, run=run or STATE["name"], pids=[p.pid for p in procs])
    name = st["run"]
    if name:
        dirs = agent_dirs(name) or [C.LOG_DIR / name]
        rows = [r for r in (last_row(d / "episodes.csv") for d in dirs) if r]
        cfg = run_config(name)
        st.update(timesteps=int(sum(float(r["timesteps"]) for r in rows) / len(rows)) if rows else 0,
                  total_steps=cfg.get("total_steps"), level=current_level(name), stage=current_stage(name),
                  agents=len(agent_dirs(name)) or None)
        try:
            lines = [l for l in (C.LOG_DIR / f"{name}.out").read_text(errors="ignore").splitlines() if "Warning" not in l and l.strip()]
            st["log"] = lines[-6:]
        except OSError:
            st["log"] = []
    if STATE["started"] and ours:
        st["elapsed"] = int(time.time() - STATE["started"])
    if proc is not None and not ours and not running:
        st["exit_code"] = proc.returncode
    return st


def start_training(params: dict) -> tuple:
    plan = schema.build_run(params)
    if "errors" in plan:
        return 400, plan
    with LOCK:
        if training_procs(fresh=True):
            return 409, dict(errors=["A training run is already running. Stop it first."])
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore")
        env["ROCKETEER_OVERRIDES"] = json.dumps(plan["overrides"])
        C.LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = open(C.LOG_DIR / f"{plan['name']}.out", "w")
        flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                 | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))   # keep the UI responsive while training
        STATE["proc"] = subprocess.Popen(plan["argv"], cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, creationflags=flags)
        STATE.update(name=plan["name"], started=time.time(), run=plan["name"])
    return 200, dict(name=plan["name"], overrides=plan["overrides"])


def stop_training() -> dict:
    ps = training_procs(fresh=True)
    for p in ps:
        kill_tree(p)
    _PROCS["t"] = 0.0
    return dict(stopped=[p.pid for p in ps])


# ------------------------------------------------------------------ data tab: export / import / clone
IMPORTS_DIR = C.DATA_DIR / "imports"
NAME_RE = re.compile(r"[A-Za-z0-9_\-]{1,60}")
MAX_UPLOAD = 400 * 1024 * 1024
FT = dict(proc=None, out=None, started=None)


def _flt(src: dict) -> dict:
    """filter fields from a query / JSON body -> keyword arguments of the data readers"""
    def num(k):
        v = src.get(k)
        v = v[0] if isinstance(v, list) else v
        return float(v) if v not in (None, "") else None

    def flag(k):
        v = src.get(k)
        v = v[0] if isinstance(v, list) else v
        return True if str(v).lower() in ("1", "true", "on", "yes") else None

    kept = src.get("kept")
    kept = kept[0] if isinstance(kept, list) else kept
    f = dict(min_merit=num("min_merit"), min_return=num("min_return"), max_return=num("max_return"), soft_only=flag("soft_only"),
             success_only=flag("success_only"), kept_for=str(kept).split("+") if kept else None)
    return {k: v for k, v in f.items() if v is not None}


def data_source(name: str):
    """a run name (trajectory data) or the name of an uploaded CSV -> what the data readers take"""
    if name.endswith((".csv", ".csv.gz")):
        f = IMPORTS_DIR / Path(name).name
        if not f.is_file():
            raise FileNotFoundError(f"no uploaded file {name}")
        return str(f)
    if not NAME_RE.fullmatch(name) or not (D.ROOT_DIR / name).is_dir():
        raise FileNotFoundError(f"no trajectory data for {name}")
    return name


def list_checkpoints() -> list:
    if not C.CHECKPOINT_DIR.exists():
        return []
    fs = sorted(C.CHECKPOINT_DIR.glob("*.zip"), key=lambda f: -f.stat().st_mtime)
    out = []
    for f in fs:
        try:
            kind = WIO.checkpoint_info(f.stem)["kind"]
        except Exception:
            kind = "unreadable"
        out.append(dict(name=f.stem, bytes=f.stat().st_size, mtime=f.stat().st_mtime, backup=".bak_" in f.name, kind=kind))
    return out


def list_imports() -> list:
    if not IMPORTS_DIR.exists():
        return []
    out = []
    for f in sorted(IMPORTS_DIR.iterdir(), key=lambda f: -f.stat().st_mtime):
        if f.is_file() and f.suffix in (".csv", ".gz"):
            with (gzip.open(f, "rt") if f.suffix == ".gz" else open(f)) as fh:
                head = fh.readline().strip()
            out.append(dict(name=f.name, bytes=f.stat().st_size, kind="weights" if head.startswith("tensor,") else "trajectories" if head.startswith("ep,t,o0") else "other"))
    return out


def data_overview() -> dict:
    return dict(runs=D.run_summaries(), checkpoints=list_checkpoints(), imports=list_imports(), used_bytes=D.total_bytes(),
                budget_bytes=int(C.DATA_BUDGET_GB * 1e9), episodes=[r for r in run_dirs() if (C.LOG_DIR / r / "episodes.csv").exists() or agent_dirs(r)])


def data_preview(source: str, flt: dict) -> dict:
    """how many logged episodes / steps pass the filters (read from the small episode tables, not the step files)"""
    src = data_source(source)
    if Path(src).is_file():
        import pandas as pd
        head = pd.read_csv(src, nrows=0).columns
        use = ["ep"] + [c for c in D.META_COLS if c in head]
        df = pd.read_csv(src, usecols=use)
        eps = 0
        steps = 0
        for _, g in df.groupby("ep", sort=False):
            meta = {c: g[c].iloc[0] for c in use[1:]}
            if D._passes(meta, flt):
                eps += 1
                steps += len(g)
        return dict(episodes=eps, steps=steps)
    eps = steps = 0
    for d in D.source_dirs(src):
        for meta in D._episode_table(d).values():
            if D._passes(meta, flt):
                eps += 1
                steps += int(meta.get("steps", 0) or 0)
    return dict(episodes=eps, steps=steps)


def send_file_attachment(handler, path: Path, filename: str, delete: bool = False) -> None:
    size = path.stat().st_size
    handler.send_response(200)
    handler.send_header("Content-Type", "text/csv; charset=utf-8")
    handler.send_header("Content-Length", str(size))
    handler.send_header("Content-Disposition", f'attachment; filename="{filename}"')
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                handler.wfile.write(chunk)
    finally:
        if delete:
            path.unlink(missing_ok=True)


def build_download(q: dict) -> tuple:
    """-> (temp or real path, download filename, delete_after)"""
    kind = q.get("kind", [""])[0]
    if kind == "episodes":
        run = q.get("run", [""])[0]
        if run not in run_dirs():
            raise FileNotFoundError(f"no run {run}")
        dirs = agent_dirs(run)
        if not dirs:
            return C.LOG_DIR / run / "episodes.csv", f"{run}_episodes.csv", False
        import pandas as pd
        tmp = Path(tempfile.mkstemp(suffix=".csv")[1])
        pd.concat([pd.read_csv(d / "episodes.csv").assign(agent=i) for i, d in enumerate(dirs)]).to_csv(tmp, index=False)
        return tmp, f"{run}_episodes_all_agents.csv", True
    if kind == "trajectories":
        src = data_source(q.get("run", [""])[0])
        tmp = Path(tempfile.mkstemp(suffix=".csv")[1])
        D.export_flat_csv(src, tmp, **_flt(q))
        return tmp, f"{Path(src).stem}_trajectories.csv", True
    if kind == "weights":
        ckpt = q.get("ckpt", [""])[0]
        if not NAME_RE.fullmatch(ckpt):
            raise FileNotFoundError("bad checkpoint name")
        tmp = Path(tempfile.mkstemp(suffix=".csv")[1])
        WIO.export_weights_csv(ckpt, tmp)
        return tmp, f"{ckpt}_weights.csv", True
    raise FileNotFoundError("unknown download kind")


def save_upload(body: dict) -> dict:
    name = re.sub(r"[^A-Za-z0-9_.\-]", "_", Path(str(body.get("name", "upload.csv"))).name)[:80]
    if not name.endswith(".csv"):
        name += ".csv"
    text = body.get("text", "")
    head = text[:200].splitlines()[0].strip() if text else ""
    if not (head.startswith("tensor,shape,index,value") or head.startswith("ep,t,o0")):
        raise ValueError("this does not look like a weights CSV (tensor,shape,index,value) or a trajectories CSV (ep,t,o0,...)")
    IMPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (IMPORTS_DIR / name).write_text(text, newline="")
    return dict(name=name, kind="weights" if head.startswith("tensor") else "trajectories", bytes=len(text))


def import_weights(body: dict) -> tuple:
    f, ckpt, out = str(body.get("file", "")), str(body.get("checkpoint", "")), str(body.get("out", "") or "")
    if not (IMPORTS_DIR / Path(f).name).is_file():
        return 404, dict(errors=[f"no uploaded file {f}"])
    if not NAME_RE.fullmatch(ckpt) or (out and not NAME_RE.fullmatch(out)):
        return 400, dict(errors=["checkpoint names: letters, digits, _ and - only"])
    if not out:
        run = train_status()
        if run.get("running") and run.get("run") and ckpt.startswith(run["run"]):
            return 409, dict(errors=["That checkpoint belongs to the run that is training right now and would be overwritten when it saves. Import into a new name instead, or stop the run first."])
    try:
        return 200, WIO.import_weights_csv(IMPORTS_DIR / Path(f).name, ckpt, out_name=out or None)
    except (ValueError, FileNotFoundError) as e:
        return 400, dict(errors=[str(e)])


def start_finetune(body: dict) -> tuple:
    if FT["proc"] is not None and FT["proc"].poll() is None:
        return 409, dict(errors=["A cloning job is already running."])
    ckpt, out, source = str(body.get("checkpoint", "")), str(body.get("out", "")), str(body.get("source", ""))
    if not NAME_RE.fullmatch(ckpt) or not (C.CHECKPOINT_DIR / f"{ckpt}.zip").exists():
        return 400, dict(errors=[f"no checkpoint named {ckpt}"])
    if not NAME_RE.fullmatch(out):
        return 400, dict(errors=["new model name: letters, digits, _ and - only (max 60)"])
    if (C.CHECKPOINT_DIR / f"{out}.zip").exists():
        return 409, dict(errors=[f"{out} already exists; pick another name (cloning never overwrites a model)"])
    try:
        src = data_source(source)
    except FileNotFoundError as e:
        return 400, dict(errors=[str(e)])
    dk, ck = D.dataset_info(src)["kind"], WIO.checkpoint_info(ckpt)["kind"]
    if dk != "unknown" and ck != "unknown" and dk != ck:
        return 400, dict(errors=[f"{ckpt} is a {ck}-lander model but {source} holds {dk}-lander data. Clone a dual model from dual data and a single one from single data "
                                 "(the Checkpoints tab makes a dual copy of a single model)."])
    flt = _flt(body)
    try:
        epochs, lr = int(float(body.get("epochs", 30))), float(body.get("lr", 1e-3))
    except ValueError:
        return 400, dict(errors=["epochs and learning rate must be numbers"])
    if not (1 <= epochs <= 500) or not (1e-6 <= lr <= 0.1):
        return 400, dict(errors=["epochs 1-500, learning rate 1e-6 to 0.1"])
    extra = []
    try:
        if body.get("log_std") not in (None, ""):
            extra += ["--log-std", str(float(body["log_std"]))]
        if body.get("low_weight") not in (None, ""):
            extra += ["--low-weight", str(float(body["low_weight"]))]
    except ValueError:
        return 400, dict(errors=["exploration log-std and the low-altitude weight must be numbers"])
    bc, value = bool(body.get("bc", True)), bool(body.get("value", True))
    if not (bc or value):
        return 400, dict(errors=["tick behaviour cloning, the value warm-up, or both"])
    argv = [sys.executable, "src/training/finetune.py", "--checkpoint", ckpt, "--data", src, "--out", out, "--epochs", str(epochs), "--lr", str(lr)]
    argv += extra
    if not bc:
        argv.append("--no-bc")
    if not value:
        argv.append("--no-value")
    for k, flag in (("min_merit", "--min-merit"), ("min_return", "--min-return"), ("max_return", "--max-return")):
        if k in flt:
            argv += [flag, str(flt[k])]
    if flt.get("soft_only"):
        argv.append("--soft-only")
    if flt.get("success_only"):
        argv.append("--success-only")
    if flt.get("kept_for"):
        argv += ["--kept", "+".join(flt["kept_for"])]
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    FT.update(proc=subprocess.Popen(argv, cwd=ROOT, env=env, stdout=open(C.LOG_DIR / f"finetune_{out}.out", "w"), stderr=subprocess.STDOUT, creationflags=flags),
              out=out, started=time.time())
    return 200, dict(out=out)


def finetune_status() -> dict:
    proc, out = FT["proc"], FT["out"]
    if out is None:
        return dict(state="none")
    running = proc is not None and proc.poll() is None
    try:
        lines = [l for l in (C.LOG_DIR / f"finetune_{out}.out").read_text(errors="ignore").splitlines() if l.strip() and "Warning" not in l]
    except OSError:
        lines = []
    res = dict(state="running" if running else "done", out=out, log=lines[-12:], elapsed=int(time.time() - FT["started"]), code=None if running else proc.returncode)
    if not running and proc.returncode == 0:
        try:
            res["report"] = json.loads((C.MODEL_CONFIG_DIR / f"{out}.json").read_text())
        except (OSError, ValueError):
            pass
    return res


def run_tests() -> None:
    t = STATE["tests"]
    if t["state"] == "running":
        return
    t.update(state="running", output="", started=time.time(), code=None)

    def work():
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider"], cwd=ROOT,
                               capture_output=True, text=True, timeout=600)
            t.update(state="done", output=(r.stdout + r.stderr)[-8000:], code=r.returncode)
        except Exception as e:
            t.update(state="done", output=f"{type(e).__name__}: {e}", code=-1)

    threading.Thread(target=work, daemon=True).start()


def live_episode(run: str, env: int, after: int, want_act: bool = False, ep_id: int | None = None) -> dict:
    """Newest streamed episode of an env newer than `after`, or the one with id `ep_id`. The neural-network activations
    (`act`, ~160 KB) are only included when `want_act` is set, i.e. for the agent selected in the Brain tab."""
    d = C.LOG_DIR / run / "live"
    files = sorted(d.glob(f"ep_*_e{env}.json")) if d.exists() else []
    if not files:
        return {"wait": True, "latest": 0}
    latest = int(files[-1].stem.split("_")[1])
    if ep_id is not None:
        files = [f for f in files if int(f.stem.split("_")[1]) == ep_id]
        after = -1
        if not files:
            return {"wait": True, "latest": latest}
    elif latest <= after:
        return {"wait": True, "latest": latest}
    for f in reversed(files):
        try:
            ep = json.loads(f.read_text())
            ep["step"], ep["live"], ep["latest"] = ep["timesteps"], True, latest
            if not want_act:
                ep.pop("act", None)
                ep.pop("actB", None)
            return ep
        except (OSError, ValueError):
            continue
    return {"wait": True, "latest": latest}


REPLAY_DIR = ROOT / "replays"            # the showcase library written by tools/record_replays.py (index.json + one file per landing)


def replay_index() -> dict:
    """The showcase library's index; empty list when nothing has been recorded yet."""
    try:
        idx = json.loads((REPLAY_DIR / "index.json").read_text())
    except (OSError, ValueError):
        return {"episodes": []}
    return idx if isinstance(idx.get("episodes"), list) else {"episodes": []}


def replay_facts() -> dict:
    """The aggregate numbers and notes the showcase interface shows (replays/facts.json); empty when it has not been recorded."""
    try:
        f = json.loads((REPLAY_DIR / "facts.json").read_text())
    except (OSError, ValueError):
        return {}
    return f if isinstance(f, dict) else {}


def replay_episode(name: str) -> bytes | None:
    """One recorded landing as the raw JSON bytes. Only files listed in the index are served (no paths from the URL)."""
    if name not in {e.get("file") for e in replay_index()["episodes"]}:
        return None
    try:
        return (REPLAY_DIR / name).read_bytes()
    except OSError:
        return None


NB = len(C.ALT_BANDS_M) + 1


class Tail:
    """Incrementally parses an episodes.csv (reads only newly appended rows) into 25k-step bins."""
    BIN = 25_000

    def __init__(self, path: Path):
        self.path, self.offset, self.head = path, 0, None
        self.bins: dict = {}
        self.recent: list = []        # last 100 strict successes
        self.recent_soft: list = []
        self.recent_merit: list = []
        self.last_t = 0

    def update(self) -> "Tail":
        try:
            size = self.path.stat().st_size
            if size < self.offset:
                self.__init__(self.path)
            if size == self.offset:
                return self
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                chunk = f.read()
        except OSError:
            return self
        end = chunk.rfind(b"\n")
        if end < 0:
            return self
        self.offset += end + 1
        for row in csv.reader(chunk[:end + 1].decode(errors="ignore").splitlines()):
            if self.head is None:
                self.head = {c: i for i, c in enumerate(row)}
                continue
            try:
                t = float(row[self.head["timesteps"]])
                ok, crash, fo = (row[self.head[k]] == "True" for k in ("success", "crashed", "out_of_fuel"))
            except (KeyError, ValueError, IndexError):
                continue
            soft = row[self.head["soft"]] == "True" if "soft" in self.head else False
            mer = float(row[self.head["merit"]]) if "merit" in self.head else 0.0
            b = self.bins.setdefault(int(t // self.BIN) * self.BIN, [0, 0, 0, 0, 0, 0.0, 0.0, 0, 0.0, 0, 0, 0.0, 0.0] + [0.0] * 2 * NB)
            b[0] += 1; b[1] += ok; b[2] += crash; b[3] += fo; b[4] += soft; b[5] += mer
            env, brk = self._num(row, "env_ok"), self._num(row, "brake_peak")          # descent-quality columns (runs started after they were added)
            if env == env:
                b[6] += env; b[7] += 1
            if brk == brk:
                b[8] += brk; b[9] += 1
            spd, dist = self._num(row, "speed_mps"), self._num(row, "dist_m")
            if spd == spd:
                b[10] += 1; b[11] += spd
            if dist == dist:
                b[12] += dist
            for i in range(NB):                                                       # speed / envelope per altitude band (learning heat-map)
                ov = self._num(row, f"ov_{i}")
                if ov == ov:
                    b[13 + i] += ov; b[13 + NB + i] += 1
            self.last_t = t
            self.recent.append(ok); self.recent_soft.append(soft); self.recent_merit.append(mer)
            del self.recent[:-100], self.recent_soft[:-100], self.recent_merit[:-100]
        return self

    def _num(self, row, name) -> float:
        try:
            return float(row[self.head[name]])
        except (KeyError, ValueError, IndexError):
            return float("nan")


TAILS: dict = {}


def tail(path: Path) -> Tail:
    return TAILS.setdefault(str(path), Tail(path)).update()


def rec(k: int, v: list) -> dict:
    n = v[0]
    return dict(t=k, n=n, success=v[1] / n, crash=v[2] / n, fuel_out=v[3] / n, soft=v[4] / n, merit=v[5] / n,
                env=v[6] / v[7] if v[7] else None,                 # share of the airborne descent inside the safe-speed envelope
                brake=v[8] / v[9] if v[9] else None,               # mean peak braking demand (1 = all the engine can do)
                td_speed=v[11] / v[10] if v[10] else None,         # mean touchdown speed over episodes that reached the ground (m/s)
                td_frac=v[10] / n,                                 # share of episodes that reached the ground
                dist=v[12] / n,                                    # mean distance from the pad centre at the end (m)
                bands=[v[13 + i] / v[13 + NB + i] if v[13 + NB + i] else None for i in range(NB)])   # mean speed / envelope per altitude band


def series(t: Tail) -> list:
    return [rec(k, v) for k, v in sorted(t.bins.items())]


def live_ids(run: str) -> dict:
    """Newest streamed episode id per env: one cheap poll tells the page which envs have something new."""
    d = C.LOG_DIR / run / "live"
    out: dict = {}
    if d.exists():
        for f in d.glob("ep_*_e*.json"):
            _, i, e = f.stem.split("_")
            k = int(e[1:])
            out[k] = max(out.get(k, 0), int(i))
    return out


def progress(run: str) -> dict:
    """Aggregate curves for the run, plus per-agent info when it is a learning-rate sweep."""
    dirs = agent_dirs(run)
    tails = [tail(d / "episodes.csv") for d in dirs] or [tail(C.LOG_DIR / run / "episodes.csv")]
    agg: dict = {}
    for t in tails:
        for k, v in t.bins.items():
            a = agg.setdefault(k, [0, 0, 0, 0, 0, 0.0, 0.0, 0, 0.0, 0, 0, 0.0, 0.0] + [0.0] * 2 * NB)
            for i in range(len(a)):
                a[i] += v[i]
    out = dict(progress=[rec(k, v) for k, v in sorted(agg.items())])
    if dirs:
        sw = run_config(run).get("sweep", {})
        lrs, grp, cp = sw.get("lrs", []), sw.get("groups", []), sw.get("copies", 1) or 1
        out["agents"] = [dict(k=i, lr=lrs[i] if i < len(lrs) else None, group=grp[i] if i < len(grp) else i, copy=i % cp, level=current_level_dir(d), timesteps=int(t.last_t),
                              recent_success=(sum(t.recent) / len(t.recent)) if t.recent else 0.0,
                              recent_soft=(sum(t.recent_soft) / len(t.recent_soft)) if t.recent_soft else 0.0,
                              recent_merit=(sum(t.recent_merit) / len(t.recent_merit)) if t.recent_merit else 0.0, series=series(t))
                         for i, (d, t) in enumerate(zip(dirs, tails))]
    return out


def current_level_dir(d: Path) -> int:
    r = last_row(d / "levels.csv")
    return int(r["level"]) if r else 0


# ------------------------------------------------------------------ http
class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):          # a page closing mid-request is not an error
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"          # keep-alive: far fewer sockets for the many small polls

    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json", code=200):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def host_ok(self) -> bool:
        return self.headers.get("Host", "") in (f"localhost:{PORT}", f"127.0.0.1:{PORT}")

    def do_GET(self):
        if not self.host_ok():
            return self.send_error(403)
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self.send(HTML.read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/cam_hulls":                                     # the landing cams' hull pictures (src/app/cam_hulls/*.png) with the height of each hull's top
                self.send(camhulls.hull_list())
            elif u.path.startswith("/models/"):
                f = camhulls.model_file(u.path[len("/models/"):])
                self.send(f.read_bytes(), "image/png") if f else self.send_error(404)
            elif u.path.startswith("/cam_hulls/"):
                f = camhulls.hull_file(u.path[len("/cam_hulls/"):])
                self.send(f.read_bytes(), "image/png") if f else self.send_error(404)
            elif u.path == "/cam_hull.png":
                f = Path(__file__).with_name("cam_hull.png")
                self.send(f.read_bytes(), "image/png") if f.exists() else self.send_error(404)
            elif u.path == "/api/schema":
                self.send(schema.public_schema())
            elif u.path == "/api/runs":
                self.send(dict(runs=run_dirs(), current=STATE["run"]))
            elif u.path == "/api/info":
                run = STATE["run"]
                n = (run_config(run).get("n_envs") if run else None) or C.N_ENVS
                self.send(dict(run=run, level=current_level(run) if run else 0, stage=current_stage(run) if run else None, n_envs=n, **(progress(run) if run else dict(progress=[]))))
            elif u.path == "/api/live":
                run = STATE["run"]
                env, after = int(q.get("env", ["0"])[0]), int(q.get("after", ["0"])[0])
                want_act = q.get("act", ["0"])[0] == "1"
                ep_id = int(q["id"][0]) if "id" in q else None
                self.send(live_episode(run, env, after, want_act, ep_id) if run else {"wait": True, "latest": 0})
            elif u.path == "/api/live_ids":
                self.send(live_ids(STATE["run"]) if STATE["run"] else {})
            elif u.path == "/api/replay/index":                                  # showcase mode: the saved landings (nothing is simulated)
                self.send(replay_index())
            elif u.path == "/api/replay/record/status":                          # the Showcase Recorder screen: progress of tools/record_replays.py
                self.send(jobs.replay_record_status())
            elif u.path == "/api/replay/facts":
                self.send(replay_facts())
            elif u.path == "/api/replay/episode":
                data = replay_episode(q.get("file", [""])[0])
                self.send(data, "application/json") if data is not None else self.send_error(404)
            elif u.path == "/api/train/status":
                self.send(train_status())
            elif u.path == "/api/data/overview":
                self.send(data_overview())
            elif u.path == "/api/data/preview":
                self.send(data_preview(q.get("source", [""])[0], _flt(q)))
            elif u.path == "/api/data/job":
                self.send(finetune_status())
            elif u.path == "/api/data/download":
                try:
                    path, fname, delete = build_download(q)
                except (FileNotFoundError, ValueError) as e:
                    return self.send({"error": str(e)}, code=404)
                return send_file_attachment(self, path, fname, delete)
            elif u.path == "/api/startup/status":
                self.send(startup.status())
            elif u.path == "/api/pipeline/status":
                self.send(jobs.pipeline_status(q.get("name", [None])[0]))
            elif u.path == "/api/pipeline/list":
                self.send(dict(pipelines=jobs.pipeline_list()))
            elif u.path == "/api/ckpt/list":
                st = train_status()
                self.send(dict(checkpoints=jobs.checkpoint_details(st.get("run") if st.get("running") else None)))
            elif u.path == "/api/ckpt/job":
                self.send(jobs.ckpt_eval_status())
            elif u.path == "/api/tests/status":
                t = dict(STATE["tests"])
                if t["started"]:
                    t["elapsed"] = int(time.time() - t["started"])
                self.send(t)
            else:
                self.send_error(404)
        except Exception as e:
            self.send({"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        if not self.host_ok() or "application/json" not in self.headers.get("Content-Type", ""):
            return self.send_error(403)
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > MAX_UPLOAD:
                return self.send({"errors": ["file too large (400 MB max)"]}, code=413)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/train/start":
                code, res = start_training(body)
                self.send(res, code=code)
            elif self.path == "/api/train/stop":
                self.send(stop_training())
            elif self.path == "/api/data/upload":
                self.send(save_upload(body))
            elif self.path == "/api/data/import_weights":
                code, res = import_weights(body)
                self.send(res, code=code)
            elif self.path == "/api/data/finetune":
                code, res = start_finetune(body)
                self.send(res, code=code)
            elif self.path == "/api/startup/begin":
                code, res = startup.begin(body)
                self.send(res, code=code)
            elif self.path == "/api/startup/stop":
                self.send(startup.stop())
            elif self.path == "/api/pipeline/start":
                code, res = jobs.pipeline_start(body)
                self.send(res, code=code)
            elif self.path == "/api/pipeline/stop":
                self.send(jobs.pipeline_stop())
            elif self.path == "/api/ckpt/make_dual":
                self.send(jobs.ckpt_make_dual(body))
            elif self.path == "/api/ckpt/make_pair":
                self.send(jobs.ckpt_make_pair(body))
            elif self.path == "/api/ckpt/quiet":
                self.send(jobs.ckpt_quiet(body))
            elif self.path == "/api/ckpt/delete":
                st = train_status()
                self.send(jobs.ckpt_delete(body, st.get("run") if st.get("running") else None))
            elif self.path == "/api/ckpt/evaluate":
                code, res = jobs.ckpt_eval_start(body)
                self.send(res, code=code)
            elif self.path == "/api/replay/record/start":
                code, res = jobs.replay_record_start(body)
                self.send(res, code=code)
            elif self.path == "/api/replay/record/stop":
                self.send(jobs.replay_record_stop())
            elif self.path == "/api/tests/run":
                run_tests()
                self.send({"ok": True})
            elif self.path == "/api/view":
                if body.get("run") in run_dirs():
                    STATE["run"] = body["run"]
                self.send({"run": STATE["run"]})
            else:
                self.send_error(404)
        except ValueError as e:                                   # bad input (including a file that is not the expected CSV)
            self.send({"errors": [str(e)]}, code=400)
        except Exception as e:
            self.send({"errors": [f"{type(e).__name__}: {e}"]}, code=500)


def main():
    global PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--run", default=None, help="run to show first (default: most recent)")
    ap.add_argument("--open", action="store_true", help="open the browser")
    a = ap.parse_args()
    PORT = a.port
    startup.HOOKS.update(start=start_training, stop=stop_training, status=train_status)
    runs = run_dirs()
    STATE["run"] = a.run or (runs[0] if runs else None)
    srv = Server(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"   # avoids the slow IPv6 "localhost" lookup on Windows
    print(f"SpaceXperiment app on {url}  (Ctrl+C to stop)", flush=True)
    if a.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
