"""Records the showcase library: finished landings of a trained controller, saved as viewer episodes, plus the facts the showcase interface shows.

    python tools/record_replays.py                      # -> replays/index.json, replays/facts.json, replays/ep_NN_L<level>_s<seed>.json
    python tools/record_replays.py --facts-only         # rewrite facts.json only (no landings are flown)

The same command runs from the browser: the "Showcase Recorder" button in the app's Interface switch (src/app/jobs.py starts this script and shows its progress).
The new library is built in replays.tmp/ and moved over replays/ only when it is complete, so a run that fails or is cancelled leaves the saved library as it was.

The Showcase interface of the web app (the third button next to Simple / Advanced, or http://127.0.0.1:8767/?showcase=1) only reads these files and draws
them. No simulator, no network and no training run while it plays, so it is almost free for the laptop.

Which landings: the checkpoint is flown again with the same seeds and the same deterministic policy as the paper's evaluation
(rules stage 2, `predict(deterministic=True)`), so every recording can be checked against the evaluation table (`--table`, default
paper_assets/retrain/episodes.csv). From that table the script picks
  * the paper's featured landing (level 1, seed 122),
  * `--per-level` typical successes at every wind level L0-L3 (the successes closest to the level's median touchdown speed and pad offset),
  * `--failures` typical failures (levels 1-3), shown honestly but never two in a row.
After flying each seed it compares the recorded touchdown speed with the table and stops if they differ (the recording is then not the evaluated landing).

Every landing carries `stats` (start state, burn, peak loads, touchdown, propellant, wind, rank among the 300 evaluated seeds of its level).
facts.json holds the aggregate numbers (success rates with 95 % Wilson intervals, failure reasons, the training curve), the rules, the level table and the
method notes, all read from paper_assets/ and src/utils/config.py (nothing is typed in twice except the training recipe, which cites its source file).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.sim.env import SpaceXEnv

DEFAULT_TABLE = ROOT / "paper_assets" / "retrain" / "episodes.csv"
SUMMARY = ROOT / "paper_assets" / "retrain" / "summary.csv"
ATTEMPTS = ROOT / "paper_assets" / "retrain" / "attempts.md"
TABLE_NAME = {"rt1_final": "sx_retrained"}                      # the controller's name in the evaluation table
FEATURED = (1, 122)                                              # (level, seed) of the landing in the paper's figures
LEVEL_ORDER = [0, 2, 1, 3]                                       # successes alternate between calm and windy
FAIL_LEVELS = [2, 3, 1]
G0 = 9.81


def load_rows(table: Path, controller: str) -> list[dict]:
    rows = [r for r in csv.DictReader(open(table)) if r["controller"] == controller]
    if not rows:
        raise SystemExit(f"no rows for {controller!r} in {table}")
    return rows


def typical(rows: list[dict], n: int, skip=()) -> list[dict]:
    """The n rows closest to the median touchdown speed and pad offset (both measured in standard deviations)."""
    rows = [r for r in rows if int(r["seed"]) not in skip]
    ms, mo = (statistics.median(float(r[k]) for r in rows) for k in ("touchdown_speed_mps", "pad_offset_m"))
    ss, so = (statistics.pstdev(float(r[k]) for r in rows) or 1.0 for k in ("touchdown_speed_mps", "pad_offset_m"))
    return sorted(rows, key=lambda r: ((float(r["touchdown_speed_mps"]) - ms) / ss) ** 2 + ((float(r["pad_offset_m"]) - mo) / so) ** 2)[:n]


def plan(rows: list[dict], per_level: int, failures: int) -> list[dict]:
    """The ordered list of table rows to record."""
    by = lambda lv, ok: [r for r in rows if int(r["level"]) == lv and (r["success"] == "1") == ok]
    feat = next(r for r in by(FEATURED[0], True) if int(r["seed"]) == FEATURED[1])
    taken = {FEATURED}
    succ = [feat]
    pools = {lv: typical(by(lv, True), per_level + 1, skip={FEATURED[1]} if lv == FEATURED[0] else ()) for lv in range(4)}
    for k in range(per_level):
        for lv in LEVEL_ORDER:
            if k == 0 and lv == FEATURED[0]:
                continue                                         # the featured landing is this level's first
            r = pools[lv].pop(0) if pools[lv] else None
            if r and (lv, int(r["seed"])) not in taken:
                taken.add((lv, int(r["seed"]))); succ.append(r)
    fails = []
    for lv in FAIL_LEVELS[:failures]:
        if by(lv, False):
            fails.append(typical(by(lv, False), 1)[0])
    out, gap = [], max(1, len(succ) // (len(fails) + 1))
    for i, r in enumerate(succ):                                 # failures go in between, never two in a row, never first
        out.append(r)
        if fails and (i + 1) % gap == 0 and i + 1 < len(succ):
            out.append(fails.pop(0))
    return out + fails


def fly(model, level: int, seed: int) -> tuple[dict, dict]:
    env = SpaceXEnv(level=level, record=True)
    env.set_rules_stage(2)
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        obs, _, te, tr, info = env.step(model.predict(obs, deterministic=True)[0])
        done = te or tr
    return info["trajectory"], info


def reason_of(row: dict, result: dict) -> str:
    if result["success"]:
        return "none"
    if row["failure_reason"] not in ("", "none"):
        return row["failure_reason"]
    return result["end"]


def episode_stats(traj: dict, info: dict, row: dict, level_rows: list[dict]) -> dict:
    """The numbers a reader asks for about one landing, measured on the recorded frames and the simulator's own episode metrics."""
    fr, dt, leg = traj["frames"], traj["dt"], traj["veh"]["leg_down"]
    f0, td, em = fr[0], info.get("touchdown") or {}, info.get("episode_metrics") or {}
    on = [i for i, f in enumerate(fr) if f.get("on", 0) > 0]
    qi = max(range(len(fr)), key=lambda i: fr[i]["q"])
    load = max(math.hypot(f["ax"], f["ay"]) for f in fr) / G0                   # the largest specific force (what the engine adds to gravity)
    burn_m = [f["m"] for f in fr if f.get("on", 0) > 0]
    pulses = lambda key: sum(abs(b.get(key, 0)) > 0.1 and not abs(a.get(key, 0)) > 0.1 for a, b in zip(fr, fr[1:]))      # thruster pulses = the times a ring switches on
    speeds = sorted(float(r["touchdown_speed_mps"]) for r in level_rows)
    offs = sorted(float(r["pad_offset_m"]) for r in level_rows)
    v = float(row["touchdown_speed_mps"])
    r1 = lambda x, n=1: None if x is None else round(float(x), n)
    return dict(
        start=dict(height_m=r1(f0["y"] - leg, 0), speed_mps=r1(math.hypot(f0["vx"], f0["vy"])), angle_deg=r1(math.degrees(math.atan2(-f0["vy"], abs(f0["vx"])))), x_offset_m=r1(abs(f0["x"]), 0),
                   tilt_deg=r1(math.degrees(abs(f0["a"])))),
        burn=dict(start_s=r1(on[0] * dt), start_alt_m=r1(fr[on[0]]["y"] - leg, 0), duration_s=r1(len(on) * dt), ignitions=int(info.get("ignitions", 0))) if on else None,
        max_q_kpa=r1(fr[qi]["q"] / 1000), max_q_alt_m=r1(fr[qi]["y"] - leg, 0), peak_load_g=r1(load),
        touchdown=dict(t_s=r1(td.get("step", len(fr)) * dt), speed_mps=r1(td.get("speed_mps"), 2), vertical_mps=r1(td.get("vspeed_mps"), 2), horizontal_mps=r1(td.get("hspeed_mps"), 2),
                       tilt_deg=r1(math.degrees(td["tilt_rad"]), 2) if td.get("tilt_rad") is not None else None, tilt_rad=r1(td.get("tilt_rad"), 4), offset_m=r1(td.get("dist_m"), 2),
                       settle_s=r1(td.get("settle_s"), 2)),
        control=dict(throttle_mean_pct=r1(100 * statistics.mean(burn_m), 0) if burn_m else None, throttle_peak_pct=r1(100 * max(burn_m), 0) if burn_m else None,
                     gimbal_max_deg=r1(math.degrees(max(abs(f.get("g", 0)) for f in fr))), fin_max_deg=r1(math.degrees(max(abs(f.get("f", 0)) for f in fr)), 0),
                     rcs_a_pulses=int(pulses("r1")), rcs_b_pulses=int(pulses("r2")), rcs_gas_left_pct=r1(100 * fr[-1].get("gas", 0), 0)),
        extremes=dict(fall_speed_mps=r1(max(-f["vy"] for f in fr), 0), sideways_speed_mps=r1(max(abs(f["vx"]) for f in fr), 0), tilt_deg=r1(math.degrees(max(abs(f["a"]) for f in fr))),
                      spin_dps=r1(math.degrees(max(abs(f["om"]) for f in fr)))),
        flight_time_s=r1(info.get("time_s")),
        fuel=dict(used_kg=r1(info.get("fuel_used"), 0), left_kg=r1(float(row["fuel_remaining_kg"]), 0), left_pct=r1(100 * info.get("fuel_remaining", 0), 0)),
        quality=dict(merit=r1(em.get("merit")), in_envelope_pct=r1(100 * em["env_ok"], 0) if em.get("env_ok") is not None else None, brake_peak_pct=r1(100 * em["brake_peak"], 0) if em.get("brake_peak") is not None else None),
        wind=dict(scale_mps=float(row["wind_scale_mps"]), mean_mps=r1(float(row["wind_mean_mps"]), 2), gust_amp_mps=r1(float(row["wind_gust_amp_mps"]), 1)),
        rank=dict(n=len(speeds), slower_than_pct=r1(100 * sum(s > v for s in speeds) / len(speeds), 0), level_median_speed_mps=r1(statistics.median(speeds), 2),
                  level_median_offset_m=r1(statistics.median(offs), 2), level_success_pct=r1(100 * sum(r["success"] == "1" for r in level_rows) / len(level_rows), 1)),
    )


# ---------------------------------------------------------------- facts.json
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def training_curve() -> list[dict]:
    """Validation success per model along the DAgger rounds, read from the table in paper_assets/retrain/attempts.md (empty when it is missing)."""
    try:
        text = ATTEMPTS.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for m in re.finditer(r"^\|\s*(\*\*)?`(rt1_(?:clone|dagger_\d))`[^|]*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(?:\*\*)?([\d.]+)", text, re.M):
        out.append(dict(name=m.group(2), levels=[float(m.group(i)) for i in (3, 4, 5, 6)], mean=float(m.group(7)), chosen="dagger_4" in m.group(2)))
    return out


def expert_validation() -> list[float]:
    """The scripted expert's validation success per level, from the first lines of attempts.md (the reference line of the training curve)."""
    try:
        m = re.search(r"scripted expert ([\d.]+) / ([\d.]+) / ([\d.]+) / ([\d.]+)", ATTEMPTS.read_text(encoding="utf-8"))
    except OSError:
        return []
    return [float(x) for x in m.groups()] if m else []


def results_block() -> dict:
    """Success rates, intervals, speeds and failure reasons per controller and level (paper_assets/retrain/summary.csv)."""
    labels = {"sx_retrained": ("Retrained network (this one)", "rt1_final, test seeds 0-299"), "expert_tuned": ("Scripted expert", "the teacher; hand-written controller"),
              "sx_imitation_quiet": ("Earlier network", "before retraining"), "pair_safe_final": ("Pair controller", "per lander; the two landers share wind")}
    out = {}
    try:
        rows = list(csv.DictReader(open(SUMMARY)))
    except OSError:
        return out
    for key, (label, note) in labels.items():
        lv = []
        for r in sorted((r for r in rows if r["controller"] == key), key=lambda r: int(r["level"])):
            lv.append(dict(level=int(r["level"]), n=int(r["n"]), successes=int(r["successes"]), rate=float(r["success_rate"]), lo=float(r["wilson_lower"]), hi=float(r["wilson_upper"]),
                           speed_median=float(r["speed_median"]), speed_mean=float(r["speed_mean"]), offset_mean=float(r["pad_offset_mean"]), tilt_mean=float(r["tilt_mean"]),
                           fuel_left_kg=float(r["fuel_remaining_mean_kg"]),
                           fails=dict(too_fast=int(r["n_too_fast"]), off_pad=int(r["n_off_pad"]), tilted=int(r["n_tilted"]), out_of_fuel=int(r["n_out_of_fuel"]), other=int(r["n_crashed_other"]))))
        if lv:
            out[key] = dict(label=label, note=note, levels=lv)
    return out


def facts(model, ck: Path, veh: dict | None) -> dict:
    from src.utils import config as C
    params = sum(p.numel() for p in model.policy.parameters())
    lv = [dict(level=i, wind_scale_mps=L["wind_power"], gust_amp_mps=round(0.4 * L["wind_power"], 1), mean_abs_wind_mps=round(0.3 * L["wind_power"], 2), speed_fraction=L["speed"],
               entry_deg=list(L["entry_deg"]), x_offset_m=L["x_offset"]) for i, L in enumerate(C.LEVELS)]
    return dict(
        version=1,
        controller=dict(name=ck.stem, file=ck.name, sha256=sha256(ck), parameters=int(params), hidden=list(C.NET_ARCH), inputs=18, outputs=5, decisions_hz=C.FPS, physics_dt_s=1.0 / (C.FPS * C.PHYSICS_SUBSTEPS)),
        rules=dict(speed_mps=C.SUCCESS_MAX_SPEED_MPS, tilt_rad=C.SUCCESS_MAX_TILT_RAD, offset_m=C.SUCCESS_MAX_DIST_M, tilt_limit_deg=C.TILT_LIMIT_DEG, max_ignitions=C.MAX_IGNITIONS, fuel_kg=C.FUEL_BUDGET,
                   spawn_height_m=C.SPAWN_HEIGHT_M, pad_half_width_m=C.PAD_HALF_WIDTH_M),
        levels=lv, vehicle=veh, results=results_block(), training_curve=training_curve(), expert_validation=expert_validation(), test_seeds=[0, 299], validation_seeds=[100000, 100199],
        recipe=dict(source="paper_assets/retrain/README.md and attempts.md", demonstrations=800, demonstrations_kept=734, clone_epochs=40, dagger_rounds=8, dagger_episodes=180, dagger_levels="0,1,1,2,2,3,3",
                    chosen_round=4, train_seconds=1015),
        generated=time.strftime("%Y-%m-%d"),
    )


def commit_library(tmp: Path, out: Path) -> None:
    """Replaces the library in `out` by the finished one in `tmp` in one quick step, so a page that reads it never sees half of each (a run that stops early leaves `out` untouched)."""
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("ep_*.json")) + [out / "index.json", out / "facts.json"]:
        old.unlink(missing_ok=True)
    for new in sorted(tmp.glob("ep_*.json")) + [tmp / "facts.json", tmp / "index.json"]:      # the index last: it is what makes the landings visible
        if new.exists():
            new.replace(out / new.name)
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", default="rt1_final", help="checkpoint name or path (models/checkpoints/<name>.zip)")
    ap.add_argument("--table", default=str(DEFAULT_TABLE), help="evaluation table (episodes.csv) used to choose seeds and to check the recording")
    ap.add_argument("--out", default=str(ROOT / "replays"))
    ap.add_argument("--per-level", type=int, default=2, help="successes per wind level (the featured landing counts for its level)")
    ap.add_argument("--failures", type=int, default=3, help="failures to include (0-3)")
    ap.add_argument("--facts-only", action="store_true", help="rewrite facts.json from the current files and fly nothing")
    a = ap.parse_args()
    from stable_baselines3 import PPO
    from src.training.weights_io import resolve_checkpoint
    ck = resolve_checkpoint(a.checkpoint)
    name = ck.stem
    model = PPO.load(str(ck), device="cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")                       # the new library is built here and moved over at the end
    shutil.rmtree(tmp, ignore_errors=True)
    if a.facts_only:
        try:
            veh = json.loads((out / next(e["file"] for e in json.loads((out / "index.json").read_text())["episodes"])).read_text())["veh"]
        except (OSError, ValueError, StopIteration, KeyError):
            veh = None
        tmp.mkdir(parents=True)
        (tmp / "facts.json").write_text(json.dumps(facts(model, ck, veh), indent=1))
        (tmp / "facts.json").replace(out / "facts.json")
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"facts.json -> {out}")
        return
    rows = load_rows(Path(a.table), TABLE_NAME.get(name, name))
    chosen = plan(rows, a.per_level, min(3, a.failures))
    tmp.mkdir(parents=True)
    print(f"plan: {len(chosen)} landings of {name}", flush=True)
    index, keep, veh = [], set(), None
    t0 = time.time()
    for n, row in enumerate(chosen, start=1):
        lv, seed = int(row["level"]), int(row["seed"])
        traj, info = fly(model, lv, seed)
        r = traj["result"]
        if abs((r.get("speed") or 0.0) - float(row["touchdown_speed_mps"])) > 5e-3 or bool(r["success"]) != (row["success"] == "1"):
            raise SystemExit(f"level {lv} seed {seed}: recorded speed {r.get('speed')} success {r['success']} does not match the table "
                             f"({float(row['touchdown_speed_mps']):.3f}, {row['success']}); is {name} the checkpoint the table was made with?")
        traj.pop("act", None)                                    # network activations (160 KB) are not needed to draw the landing
        veh = veh or traj["veh"]
        meta = dict(level=lv, seed=seed, checkpoint=name, wind_scale_mps=float(row["wind_scale_mps"]), wind_mean_mps=round(float(row["wind_mean_mps"]), 2),
                    success=bool(r["success"]), end=r["end"], reason=reason_of(row, r), speed=r.get("speed"), offset_m=r.get("dist"),
                    featured=(lv, seed) == FEATURED)
        meta["stats"] = episode_stats(traj, info, row, [x for x in rows if int(x["level"]) == lv])
        traj.update(id=n, env=0, timesteps=0, replay=meta)
        fn = f"ep_{n:02d}_L{lv}_s{seed}.json"
        (tmp / fn).write_text(json.dumps(traj, separators=(",", ":")))
        keep.add(fn)
        index.append(dict(file=fn, frames=len(traj["frames"]), seconds=round(len(traj["frames"]) * traj["dt"], 1), **meta))
        print(f"{n:2d}/{len(chosen)}  level {lv} seed {seed:3d}  {'success' if r['success'] else 'FAILED ' + meta['reason']:<14} {r.get('speed')} m/s  {r.get('dist')} m  {len(traj['frames'])} frames", flush=True)
    (tmp / "index.json").write_text(json.dumps(dict(version=2, checkpoint=name, rules_stage=2, recorded=time.strftime("%Y-%m-%d %H:%M"), episodes=index), indent=1))
    (tmp / "facts.json").write_text(json.dumps(facts(model, ck, veh), indent=1))
    kb = sum((tmp / f).stat().st_size for f in keep) / 1024
    commit_library(tmp, out)                                     # a re-run replaces the whole library
    print(f"{len(index)} episodes ({kb:.0f} KB) -> {out}  [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
