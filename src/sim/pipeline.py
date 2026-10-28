"""The imitation pipeline: expert demonstrations -> behaviour cloning -> DAgger rounds -> evaluation, for one lander or Dual land.

    python -m src.sim.pipeline --config logs/pipeline/<name>/config.json          (what the app's Pipeline tab runs)
    python -m src.sim.pipeline --name myrun [--dual | --pair] [--from-single sx_imitation_quiet] [--rounds 6] ...

Everything it makes is named after the pipeline:
    data/trajectories/<name>_expert, <name>_dagger_<r>      demonstrations and DAgger labels
    models/checkpoints/<name>_init, <name>_clone, <name>_dagger_<r>, <name>_final
<name>_final is the best DAgger round (by the mean success rate over the evaluated levels), or the clone / init if those steps were not run.
Progress is written to logs/pipeline/<name>/status.json after every event, so a page can follow it.

Pair (--pair): Dual land with TWO networks. The same pairs of experts fly, but each lander's own view and actions go to its own data set (<name>_expert_A / _B,
<name>_dagger_N_A / _B) and each network is cloned and DAgger-retrained on its own data; checkpoints <name>_clone, <name>_dagger_N, <name>_final are pairs
(files ..._A and ..._B). --from-single starts both networks as copies of a single-lander checkpoint.

Dual land: the demonstrations come from one expert per lander (they start on the same path, so they land together), the network is the
512 x 512 dual one, and --from-single starts it as two copies of a single-lander checkpoint (otherwise it is random).
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import config as C

STEP_LABELS = dict(expert="Expert demonstrations", clone="Behaviour cloning", dagger="DAgger rounds", eval="Final evaluation")
STEP_ORDER = ["expert", "clone", "dagger", "eval"]

DEFAULTS = dict(
    name="pipe", dual=False, pair=False, steps=list(STEP_ORDER), init="", from_single="", seed=0, overwrite=False,
    expert=dict(episodes=400, noise=0.12, levels="0,0,1,1,2"),
    clone=dict(epochs=40, lr=1e-3, batch=512, log_std=-1.6, low_weight=5.0),
    dagger=dict(rounds=6, episodes=160, epochs=30, lr=5e-4, batch=512, noise=0.1, levels="0,0,1,1,2", log_std=-1.6, low_weight=5.0),
    eval=dict(episodes=30, levels="0,1,2"),
)


def merged(cfg: dict) -> dict:
    out = copy.deepcopy(DEFAULTS)
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def levels_of(s) -> tuple:
    return tuple(int(x) for x in str(s).split(",") if str(x).strip() != "")


def outputs_of(name: str) -> dict:
    """What a pipeline of this name owns (to detect clashes and to clear them on request)."""
    ck = sorted(C.CHECKPOINT_DIR.glob(f"{name}_*.zip")) if C.CHECKPOINT_DIR.exists() else []
    traj = C.DATA_DIR / "trajectories"
    dd = sorted(d for d in traj.glob(f"{name}_*") if d.is_dir()) if traj.exists() else []
    return dict(checkpoints=ck, data=dd)


def clear_outputs(name: str) -> None:
    o = outputs_of(name)
    for f in o["checkpoints"]:
        f.unlink(missing_ok=True)
    for d in o["data"]:
        shutil.rmtree(d, ignore_errors=True)
    for d in C.LOG_DIR.glob(f"{name}_*"):
        shutil.rmtree(d, ignore_errors=True) if d.is_dir() else d.unlink(missing_ok=True)


class Reporter:
    """status.json for the page + a plain log."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dir = C.LOG_DIR / "pipeline" / cfg["name"]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.status = dict(name=cfg["name"], dual=bool(cfg["dual"]), state="running", started=self.t0, finished=None, error=None, config=cfg,
                           steps=[dict(key=k, label=STEP_LABELS[k], state="pending" if k in cfg["steps"] else "skipped", detail="", t0=None, t1=None) for k in STEP_ORDER],
                           rounds=[], best=None, final=None, final_eval=None)
        self.lines: list = []
        self.write()

    def step(self, key):
        return next(s for s in self.status["steps"] if s["key"] == key)

    def start(self, key):
        s = self.step(key)
        s.update(state="running", t0=time.time(), detail="")
        self.write()

    def done(self, key, detail=None):
        s = self.step(key)
        s.update(state="done", t1=time.time())
        if detail is not None:
            s["detail"] = detail
        self.write()

    def detail(self, key, text):
        self.step(key)["detail"] = text
        self.write()

    def log(self, text):
        line = f"[{time.strftime('%H:%M:%S')}] {text}"
        print(line, flush=True)
        self.lines.append(line)
        self.lines = self.lines[-60:]
        self.write()

    def write(self):
        self.status["updated"] = time.time()
        self.status["log"] = self.lines
        tmp = self.dir / "status.tmp"
        tmp.write_text(json.dumps(self.status, default=str))
        tmp.replace(self.dir / "status.json")

    def finish(self, state, error=None):
        self.status.update(state=state, finished=time.time(), error=error)
        self.write()


def run(cfg: dict, rep: Reporter | None = None) -> dict:
    from src.sim.evaluate import evaluate
    from src.sim.expert_data import collect, dagger_collect, make_init
    from src.training.finetune import finetune
    from src.training.pair import resolve_pair
    cfg = merged(cfg)
    pair = bool(cfg["pair"])
    dual, name = bool(cfg["dual"]) and not pair, cfg["name"]
    two = dual or pair                                             # two landers in the episode (one or two networks)
    kind = "pair" if pair else "dual" if dual else "single"
    cfg["dual"] = two
    rep = rep or Reporter(cfg)
    ck = lambda n: C.CHECKPOINT_DIR / f"{n}.zip"
    have = lambda n: (resolve_pair(n) is not None) if pair else ck(n).exists()          # does this checkpoint (a pair: both networks) exist
    sides = "AB" if pair else [""]                                                       # a pair is two files / two data sets: <x>_A and <x>_B
    nm = lambda base, s: f"{base}_{s}" if s else base
    steps = [k for k in STEP_ORDER if k in cfg["steps"]]
    data_expert = f"{name}_expert"
    expert_dirs = lambda: all((C.DATA_DIR / "trajectories" / nm(data_expert, s)).is_dir() for s in sides)
    cur = None                                                     # the checkpoint the next step starts from
    try:
        # ---- 1. expert demonstrations
        if "expert" in steps:
            rep.start("expert")
            e = cfg["expert"]
            res = collect(int(e["episodes"]), data_expert, levels_of(e["levels"]), float(e["noise"]), int(cfg["seed"]), dual=dual, pair=pair,
                          log=lambda s: (rep.log(s), rep.detail("expert", s.replace("[expert] ", ""))))
            rep.done("expert", f"{res['kept']} {'pairs' if two else 'landings'} kept of {res['tried']} flown, median fuel {res['median_fuel_kg'] / 1000:.1f} t"
                               + (f", median landing gap {res['median_gap_s']:.2f} s" if two else ""))
        # ---- the network(s) to start from
        start_ckpt = cfg["init"] or (f"{name}_init" if ("clone" in steps or "dagger" in steps) else "")
        if "clone" in steps or ("dagger" in steps and not have(f"{name}_clone")):
            if not cfg["init"]:
                make_init(f"{name}_init", dual=dual, pair=pair, from_single=cfg["from_single"] or None)
                rep.log(f"made {name}_init" + (".zip" if not pair else " (_A and _B)") + (f" from {'two copies' if dual else 'copies'} of {cfg['from_single']}" if two and cfg["from_single"] else " (untrained)"))
        # ---- 2. behaviour cloning (a pair: each network on its own lander's data)
        if "clone" in steps:
            rep.start("clone")
            c = cfg["clone"]
            if not expert_dirs():
                raise FileNotFoundError(f"no demonstrations named {data_expert}: run the expert step first")
            for s in sides:
                finetune(nm(start_ckpt, s), [nm(data_expert, s)], nm(f"{name}_clone", s), epochs=int(c["epochs"]), lr=float(c["lr"]), batch=int(c["batch"]),
                         log_std=float(c["log_std"]), low_weight=float(c["low_weight"]),
                         log=lambda x, s=s: (rep.log((f"[{s}] " if s else "") + x), rep.detail("clone", ((f"network {s}: " if s else "") + x)[:140])))
            cur = f"{name}_clone"
            rep.done("clone", "saved " + cur + (" (_A and _B)" if pair else ""))
        # ---- 3. DAgger rounds
        if "dagger" in steps:
            rep.start("dagger")
            d, ev = cfg["dagger"], cfg["eval"]
            if cur is None:
                cur = f"{name}_clone" if have(f"{name}_clone") else start_ckpt
            if not cur:
                raise FileNotFoundError("nothing to start DAgger from: run the clone step, or give a starting checkpoint")
            if not expert_dirs():
                raise FileNotFoundError(f"no demonstrations named {data_expert}: run the expert step first")
            data = {s: [nm(data_expert, s)] for s in sides}
            best_score = -1.0
            for r in range(1, int(d["rounds"]) + 1):
                out = f"{name}_dagger_{r}"
                rep.detail("dagger", f"round {r}/{d['rounds']}: flying {d['episodes']} episodes with {cur} while the expert labels every state")
                rep.log(f"round {r}: DAgger data from {cur}")
                ends = dagger_collect(cur, int(d["episodes"]), out, levels_of(d["levels"]), float(d["noise"]), seed=int(cfg["seed"]) + r, log=rep.log)
                for s in sides:
                    data[s].append(nm(out, s))
                    rep.detail("dagger", f"round {r}/{d['rounds']}: retraining{f' network {s}' if s else ''} on {len(data[s])} data sets")
                    finetune(nm(cur, s), data[s], nm(out, s), epochs=int(d["epochs"]), lr=float(d["lr"]), batch=int(d["batch"]), log_std=float(d["log_std"]),
                             low_weight=float(d["low_weight"]), log=lambda x, s=s: rep.log((f"[{s}] " if s else "") + x))
                cur = out
                rep.detail("dagger", f"round {r}/{d['rounds']}: evaluating {out}")
                res = evaluate(out, levels_of(ev["levels"]), int(ev["episodes"]), log=rep.log)
                rep.status["rounds"].append(dict(round=r, checkpoint=out, score=res["score"], flown=ends, levels=res["levels"]))
                if res["score"] >= best_score:
                    best_score, rep.status["best"] = res["score"], out
                rep.write()
            rep.done("dagger", f"{d['rounds']} rounds, best {rep.status['best']} (mean success {best_score:.0%})")
            cur = rep.status["best"]
        # ---- 4. final checkpoint + evaluation
        final_src = cur or (f"{name}_clone" if have(f"{name}_clone") else start_ckpt)
        if final_src:
            for s in sides:
                shutil.copy2(ck(nm(final_src, s)), ck(nm(f"{name}_final", s)))
            rep.status["final"] = f"{name}_final"
            rep.log(f"{name}_final" + (" (_A and _B)" if pair else ".zip") + f" = {final_src}")
        if "eval" in steps:
            rep.start("eval")
            if not final_src:
                raise FileNotFoundError("nothing to evaluate")
            ev = cfg["eval"]
            same = [r for r in rep.status["rounds"] if r["checkpoint"] == final_src and [x["level"] for x in r["levels"]] == list(levels_of(ev["levels"]))
                    and all(x["n"] == int(ev["episodes"]) for x in r["levels"])]
            if same:                                               # the best round was just evaluated with these very settings: reuse it
                res = dict(checkpoint=f"{name}_final", kind=kind, episodes=int(ev["episodes"]), levels=same[-1]["levels"], score=same[-1]["score"])
                rep.log(f"final evaluation = round {same[-1]['round']}'s (same settings)")
            else:
                res = evaluate(f"{name}_final", levels_of(ev["levels"]), int(ev["episodes"]), log=rep.log)
            (C.MODEL_CONFIG_DIR).mkdir(parents=True, exist_ok=True)
            (C.MODEL_CONFIG_DIR / f"{name}_final.eval.json").write_text(json.dumps(res, indent=1))
            rep.status["final_eval"] = res
            rep.done("eval", f"mean success {res['score']:.0%} over levels {ev['levels']}")
        rep.finish("done")
    except Exception as e:                                          # the page shows what went wrong
        rep.log(f"FAILED: {type(e).__name__}: {e}")
        rep.log(traceback.format_exc()[-600:])
        for s in rep.status["steps"]:
            if s["state"] == "running":
                s.update(state="failed", t1=time.time())
        rep.finish("failed", f"{type(e).__name__}: {e}")
        raise SystemExit(1)
    return rep.status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="a JSON file with the settings (what the app writes)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--dual", action="store_true", help="Dual land with ONE network flying both landers")
    ap.add_argument("--pair", action="store_true", help="Dual land with TWO networks, one per lander (own data, own cloning, own DAgger)")
    ap.add_argument("--from-single", default="")
    ap.add_argument("--init", default="", help="an existing checkpoint to clone into instead of a fresh network")
    ap.add_argument("--steps", default=",".join(STEP_ORDER))
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None, help="expert demonstrations")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()) if a.config else {}
    for k, v in (("name", a.name), ("init", a.init or None), ("from_single", a.from_single or None)):
        if v:
            cfg[k] = v
    if a.dual:
        cfg["dual"] = True
    if a.pair:
        cfg["pair"] = True
    if a.overwrite:
        cfg["overwrite"] = True
    if not a.config:
        cfg["steps"] = [s for s in a.steps.split(",") if s]
    if a.rounds is not None:
        cfg.setdefault("dagger", {})["rounds"] = a.rounds
    if a.episodes is not None:
        cfg.setdefault("expert", {})["episodes"] = a.episodes
    cfg = merged(cfg)
    if cfg["overwrite"]:
        clear_outputs(cfg["name"])
    elif any(outputs_of(cfg["name"]).values()) and "expert" in cfg["steps"]:
        sys.exit(f"{cfg['name']}_* already exists (checkpoints or data); pick another name or pass --overwrite")
    run(cfg)


if __name__ == "__main__":
    main()
