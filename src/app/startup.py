"""The Startup tab's backend: ONE call that gives the landers a head start and (optionally) goes on to train them.

    startup_begin({mode, source, from_model, quality, then_train, train: {...}})

mode    single | dual (two landers, one shared network) | pair (two landers, two networks)
source  copy   : copy a single-lander network you already have (instant); for dual / pair it is turned into a dual network / a pair
        expert : run the expert pipeline (scripted expert -> clone -> DAgger -> evaluate); with a starting network the clone step is skipped
        blank  : no head start at all
quality 1 quick / 2 standard / 3 thorough (how many demonstrations and DAgger rounds)

If then_train is set, training starts by itself when the head start is done (a small watcher thread; the page does not have to stay open).
The state lives in logs/startup.json so the page can pick it up again after a refresh.
"""
from __future__ import annotations

import json
import threading
import time

from src.app import jobs
from src.utils import config as C

HOOKS = dict(start=None, stop=None, status=None)      # filled in by the server: start_training(params) -> (code, result), stop_training(), train_status()
LOCK = threading.Lock()
PROFILE = dict(single="spacex", dual="dual", pair="pair")
LABEL = dict(single="one lander", dual="two landers, one shared network", pair="two landers, two networks")

PRESETS = {
    1: dict(label="Quick", expert=dict(episodes=150), clone=dict(epochs=25), dagger=dict(rounds=2, episodes=80, epochs=15), eval=dict(episodes=15, levels="0,1")),
    2: dict(label="Standard", expert=dict(episodes=300), clone=dict(epochs=30), dagger=dict(rounds=3, episodes=120, epochs=25), eval=dict(episodes=20, levels="0,1,2")),
    3: dict(label="Thorough", expert=dict(episodes=400), clone=dict(epochs=40), dagger=dict(rounds=6, episodes=160, epochs=30), eval=dict(episodes=30, levels="0,1,2")),
}
ETA_MIN = {1: 12, 2: 35, 3: 80}                          # rough minutes for one lander; two landers take about 1.5 times as long


def _rec_path():
    return C.LOG_DIR / "startup.json"


def _read() -> dict | None:
    try:
        return json.loads(_rec_path().read_text())
    except (OSError, ValueError):
        return None


def _write(rec: dict) -> None:
    with LOCK:
        C.LOG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _rec_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, default=str))
        tmp.replace(_rec_path())


def _update(**kw) -> None:
    rec = _read() or {}
    rec.update(kw, updated=time.time())
    _write(rec)


def _best_single() -> str | None:
    """A sensible default head start: a single-lander network that has been evaluated well, else the newest single-lander network."""
    rows = [r for r in jobs.checkpoint_details() if r["kind"] == "single" and not r["backup"]]
    scored = [r for r in rows if r.get("eval") and r["eval"].get("score") is not None]
    if scored:
        return max(scored, key=lambda r: (r["eval"]["score"], r["mtime"]))["name"]
    return rows[0]["name"] if rows else None


def begin(body: dict) -> tuple:
    mode = str(body.get("mode", "pair"))
    source = str(body.get("source", "copy"))
    if mode not in PROFILE or source not in ("copy", "expert", "blank"):
        raise ValueError("mode: single / dual / pair; source: copy / expert / blank")
    quality = jobs._num(body, "quality", 1, 3, 2, True)
    then_train = bool(body.get("then_train", True))
    tr = body.get("train") or {}
    train = dict(total_steps=jobs._num(tr, "total_steps", 50_000, 10_000_000, 1_000_000, True), n_envs=jobs._num(tr, "n_envs", 1, 16, 4, True),
                 learning_rate=jobs._num(tr, "learning_rate", 1e-6, 1e-3, 1e-4), start_level=jobs._num(tr, "start_level", 0, C.N_LEVELS - 1, 0, True))
    dual_settings = {}
    for key, field, lo, hi in (("spacing", "dual_spacing", 85, 6000), ("height", "dual_height", 500, 8000), ("jitter", "dual_jitter", 0, 1), ("tol", "dual_tol", 0.1, 20)):
        if (body.get("dual") or {}).get(key) is not None:
            dual_settings[field] = jobs._num(body["dual"], key, lo, hi, 0)
    entry = body.get("entry") or {}
    entry_preset = str(entry.get("preset", "normal"))
    if entry_preset not in ("extreme", "intermediate", "SpaceX", "normal", "straight down"):
        raise ValueError("entry angle: extreme / intermediate / SpaceX / normal / straight down")
    traj_variation = jobs._num(entry, "variation", 0, 1, 1.0)
    model = str(body.get("from_model", "") or "")
    if model and (not jobs.NAME_RE.fullmatch(model) or not jobs._exists(model)):
        raise ValueError(f"no model named {model}")
    kind = jobs._kind(model) if model else ""
    if kind == "pair":                                                       # a pair you already have (<name>_A and <name>_B) can be the head start of a two-brain run, as it is
        if not (mode == "pair" and source == "copy"):
            raise ValueError(f"{model} is a pair of brains: it can only be used as it is, as the head start of a run with two separate brains (not for one lander, one shared brain, or the expert pipeline)")
    elif model and kind != "single":
        raise ValueError(f"{model} is not a single-lander model (the head start copies a single-lander model)")
    if source == "copy" and not model:
        model = _best_single() or ""
        if not model:
            raise ValueError("there is no single-lander model to copy yet: choose 'Teach them with the expert' instead")
    if jobs._running(jobs.PIPE):
        return 409, dict(errors=[f"The pipeline {jobs.PIPE['name']} is still running. Stop it first (Stop everything)."])
    if then_train and HOOKS["status"] and HOOKS["status"]().get("running"):
        return 409, dict(errors=["A training run is already running (possibly from another project). Stop it first, or untick 'start training right after'."])
    name = str(body.get("name", "") or "").strip() or f"start_{mode}_{time.strftime('%d%H%M%S')}"
    if not jobs.NAME_RE.fullmatch(name):
        raise ValueError("name: letters, digits, _ and - only")
    overwrite = bool(body.get("overwrite", False))
    rec = dict(name=name, mode=mode, source=source, from_model=model, quality=quality, then_train=then_train, train=train, dual=dual_settings, entry=dict(preset=entry_preset, variation=traj_variation), started=time.time(), pipeline=None,
               train_started=False, train_run=None, error=None, head=dict(state="pending", detail=""), final=None)
    init_for_train = ""
    chain = False
    if source == "copy":
        if mode == "single":
            init_for_train = model
            rec["head"] = dict(state="done", detail=f"uses {model} as it is")
        elif mode == "dual":
            from src.sim.dual_init import make_dual
            if overwrite:
                (C.CHECKPOINT_DIR / f"{name}_final.zip").unlink(missing_ok=True)
            make_dual(model, f"{name}_final")
            init_for_train, rec["final"] = f"{name}_final", f"{name}_final"
            rec["head"] = dict(state="done", detail=f"two copies of {model} side by side (one dual network)")
        elif kind == "pair":
            init_for_train = model
            rec["head"] = dict(state="done", detail=f"uses the pair {model} (two networks) as it is")
        else:
            from src.training.pair import make_pair
            make_pair(model, f"{name}_final")
            init_for_train, rec["final"] = f"{name}_final", f"{name}_final"
            rec["head"] = dict(state="done", detail=f"{name}_final_A and {name}_final_B: two networks, each a copy of {model}")
    elif source == "blank":
        rec["head"] = dict(state="skipped", detail="no head start: the networks start with random weights")
    else:                                                                    # the expert pipeline
        steps = ["expert", "dagger", "eval"] if model else ["expert", "clone", "dagger", "eval"]
        p = PRESETS[quality]
        body_p = dict(name=name, dual=mode == "dual", pair=mode == "pair", steps=steps, overwrite=overwrite, expert=p["expert"], clone=p["clone"], dagger=p["dagger"], eval=p["eval"],
                      init=model if mode == "single" else "", from_single=model if mode != "single" else "")
        cfg_over = {"ENTRY_PRESET": entry_preset, "TRAJ_VARIATION": traj_variation}        # the demonstrations must come from the same kind of start as the training
        if dual_settings and mode != "single":
            keymap = dict(dual_spacing="DUAL_PAD_SPACING_M", dual_height="DUAL_SPAWN_HEIGHT_M", dual_jitter="DUAL_JITTER", dual_tol="DUAL_SYNC_TOL_S")
            cfg_over.update({keymap[k]: v for k, v in dual_settings.items()})
        env_extra = {"ROCKETEER_OVERRIDES": json.dumps({"config": cfg_over})}
        code, res = jobs.pipeline_start(body_p, env_extra=env_extra)
        if code != 200:
            return code, res
        rec["pipeline"], rec["final"] = name, f"{name}_final"
        rec["head"] = dict(state="running", detail="the expert pipeline is running")
        init_for_train, chain = f"{name}_final", True
    rec["init_for_train"] = init_for_train
    _write(rec)
    if then_train:
        params = _train_params(mode, train, dict(dual_settings, entry_preset=entry_preset, traj_variation=traj_variation), init_for_train if source != "blank" else "")
        if chain:
            threading.Thread(target=_chain, args=(name, params), daemon=True).start()
        else:
            _launch(params)
    return 200, dict(name=name, mode=mode, source=source, pipeline=rec["pipeline"], then_train=then_train, eta_min=_eta(mode, source, quality))


def _eta(mode: str, source: str, quality: int) -> int:
    if source != "expert":
        return 0
    return int(round(ETA_MIN[quality] * (1.0 if mode == "single" else 1.5)))


def _train_params(mode, train, dual_settings, init) -> dict:
    p = dict(start_profile=PROFILE[mode], n_envs=train["n_envs"], total_steps=train["total_steps"], learning_rate=train["learning_rate"], start_level=train["start_level"],
             ent_coef=0, staged="on", sweep="off", **dual_settings)
    if init:
        p["init_from"] = init
    return p


def _launch(params: dict) -> None:
    code, res = HOOKS["start"](params)
    if code == 200:
        _update(train_started=True, train_run=res["name"], train_params=params)
    else:
        _update(error="training did not start: " + " ".join(res.get("errors", [str(res)])))


def _chain(name: str, params: dict) -> None:
    """Waits for the pipeline to finish, then starts the training from its final network."""
    while jobs._running(jobs.PIPE) and jobs.PIPE["name"] == name:
        time.sleep(2.0)
    st = jobs._read_status(name)
    if st and st.get("state") == "done" and st.get("final"):
        _launch(dict(params, init_from=st["final"]))
    else:
        _update(error="the head start did not finish, so training was not started (see the pipeline log)")


def stop() -> dict:
    out = dict(pipeline=jobs.pipeline_stop())
    if HOOKS["stop"]:
        out["training"] = HOOKS["stop"]()
    rec = _read()
    if rec:
        _update(error=rec.get("error") or "stopped", stopped=True)
    return out


def status() -> dict:
    rec = _read()
    best = _best_single()
    if not rec:
        return dict(state="none", suggested_model=best, presets={k: dict(label=v["label"], eta=ETA_MIN[k]) for k, v in PRESETS.items()})
    steps = []
    head = dict(rec["head"])
    pipe = jobs.pipeline_status(rec["pipeline"]) if rec.get("pipeline") else None
    if pipe and pipe.get("state") != "none":
        run_step = next((s for s in pipe.get("steps", []) if s["state"] == "running"), None)
        done = sum(1 for s in pipe.get("steps", []) if s["state"] == "done")
        total = sum(1 for s in pipe.get("steps", []) if s["state"] != "skipped")
        st = pipe["state"]
        head = dict(state={"running": "running", "done": "done", "failed": "failed", "stopped": "failed", "starting": "running"}.get(st, "running"),
                    detail=(f"{run_step['label']}: {run_step['detail']}" if run_step else (f"finished: {pipe.get('final')}" if st == "done" else st)),
                    progress=f"{done}/{total} steps", rounds=pipe.get("rounds", []), final_eval=pipe.get("final_eval"), log=(pipe.get("log") or [])[-6:])
    steps.append(dict(key="head", label="Head start", **head))
    if rec.get("then_train"):
        train = dict(state="pending", detail="starts when the head start is done" if rec.get("pipeline") else "starting")
        if rec.get("train_started"):
            ts = HOOKS["status"]() if HOOKS["status"] else {}
            mine = ts.get("run") == rec.get("train_run")
            if ts.get("running") and mine:
                train = dict(state="running", detail=f"run {rec['train_run']}: {ts.get('timesteps', 0):,} of {ts.get('total_steps', 0):,} steps, level {ts.get('level')}",
                             progress=f"{100 * ts.get('timesteps', 0) / max(ts.get('total_steps', 1), 1):.0f}%")
            else:
                train = dict(state="done", detail=f"run {rec['train_run']} has ended (or was stopped)")
        elif rec.get("error"):
            train = dict(state="failed", detail=rec["error"])
        steps.append(dict(key="train", label="Training", **train))
    state = "failed" if any(s["state"] == "failed" for s in steps) else "running" if any(s["state"] == "running" for s in steps) else "done" if all(s["state"] in ("done", "skipped") for s in steps) else "waiting"
    return dict(state=state, name=rec["name"], mode=rec["mode"], source=rec["source"], from_model=rec.get("from_model"), final=rec.get("final"), error=rec.get("error"),
                started=rec["started"], steps=steps, train_run=rec.get("train_run"), suggested_model=best, presets={k: dict(label=v["label"], eta=ETA_MIN[k]) for k, v in PRESETS.items()})
