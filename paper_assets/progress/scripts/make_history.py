"""file_history.txt, milestones.md and stats.txt from what is on disk (there is NO git repository: dates are file-system creation (ctime) and modification (mtime) times).

    python paper_assets/progress/scripts/make_history.py

Windows reports ctime as the creation time. Files that were copied into SpaceXperiment/ at 2026-10-19 21:02 (the folder's creation) show that copy time as ctime and keep
the original modification time, so for those the mtime is the better date; the listing prints both.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

PROTO = Path("C:/Users/firdo/rocketeer")
SX = PROTO / "SpaceXperiment"
OUT = Path(__file__).resolve().parents[1]
fmt = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t))

# (date, evidence file(s), what changed [quote], what can be seen / measured)
MILESTONES = [
    ("2026-10-18 13:14", "src/training/env.py, requirements.txt, tests/test_env.py (Rocketeer folder; .gitignore 12:57)",
     'First environment (the file as it stands now; it was edited up to 17:17 that day): "Stack: LunarLander-v3 (continuous, TimeLimit 1000) -> RandomStart -> FuelBudget -> IgnitionLimit -> TouchdownLock -> TiltLimit -> StallLimit" (a gymnasium Box2D lunar lander wrapped with our own rules); `.gitignore` header "Gitignore for Rocketeer 01".',
     "A runnable 2-D lander environment with fuel, ignition, tilt and stall rules; tests/test_env.py (5.6 kB)."),
    ("2026-10-18 13:52", "src/training/train.py (13:42), src/utils/record_gif.py (13:51), assets/custom_l0_m2.gif, assets/custom_l0_m2_progress.png",
     'First PPO training script ("python src/training/train.py --reward custom|default --seed N") and GIF recorder.',
     "Real recordings: custom_l0_m2.gif (631 frames, 600x400) and a progress plot; run custom_s0_m2 (3,906 episodes): 0 strict successes, median touchdown speed falls from about 9 to about 2.5 m/s over 500k steps (train_custom_s0_m2.png)."),
    ("2026-10-18 14:34", "src/utils/config.py (14:34), src/training/calibrate_fuel.py (14:19)",
     'Difficulty levels: `LEVELS = [dict(wind_power=0.0 ... ), dict(wind_power=5.0, turbulence_power=0.5 ...), dict(wind_power=10.0 ...), dict(wind_power=15.0, turbulence_power=1.5 ...)]` and fuel calibration ("median fuel on successful landings -> budget = 1.5x"). src/training/wrappers.py sets `u.enable_wind = cfg["wind_power"] > 0.0`.',
     "Four levels L0-L3 with wind and turbulence; a calibrated fuel budget."),
    ("2026-10-18 14:47", "run_app.ps1 (14:41), src/app/server.py (14:47), src/training/sweep.py (14:47), src/app/index.html",
     'Browser control panel on port 8765 ("Rocketeer app") and a learning-rate sweep ("one independent agent per environment, learning rate growing exponentially").',
     "A live viewer of training episodes (img_02 shows the prototype viewer as it stood on 19 Oct), start/stop of training, tests tab."),
    ("2026-10-18 15:01", "src/utils/spacex_profile.py (15:00), tests/test_spacex.py (15:01)",
     '"Real Falcon 9 booster descent figures -> simulator start conditions." with the note "(all approximate / educational; no official SpaceX telemetry was found)".',
     "Start conditions taken from published descent figures instead of a gentle lunar-lander start."),
    ("2026-10-18 17:50", "src/training/reward.py (15:59), merit.py (15:59), REWARD.md (16:02), tests/test_tilt.py (16:47), tests/test_rules.py (17:18), src/training/curriculum.py (17:50)",
     'Custom reward v2 ("potential-based shaping + survive-gated landing score"), a 0-100 merit score, and a rules-stage gate ("decides when the newer environment rules switch on").',
     "Run custom_s0_staged8 (10,627 episodes, 8 environments, 18 Oct 20:24): soft landings reach about 10 % of episodes, strict success stays near 0 (train_custom_s0_staged8.png)."),
    ("2026-10-19 07:04", "src/training/brain.py (07:04), RUNBOOK.md (10:31), tests/test_progress.py (10:42)",
     '"Neural-network \'brain view\' data: what the policy network computed at every step of an episode."',
     "The viewer can show the network's activations next to the flying lander."),
    ("2026-10-19 16:38", "src/training/data.py (16:37), weights_io.py (16:37), finetune.py (16:38), tests/test_data.py (16:38), src/app/index.html (18:03 last change before the copy)",
     'Experience logging and "Train a checkpoint further from logged experience (behaviour cloning + value warm-up)". By 18:03 the prototype page already contains the telemetry, landing/nadir cam, descent phase portrait, flight recorder, decision boundary, fuel profile, landing scatter and flight display panels (ids tele, camBox, phaseBox, gp_rec, gp_bnd, gp_fuel, gp_land, gp_pfd); cam2Box (partner cam), Dual land, Pipeline and the airflow picture are not in that page.',
     "First imitation tooling; the viewer panels listed exist in the 19 Oct page (their individual creation times cannot be recovered from a single file)."),
    ("2026-10-19 21:06", "SpaceXperiment/src/sim/physics.py and vehicle.py (21:06), tests/test_sim_physics.py (21:07), folder created 21:02",
     '"2-D rigid-body physics of the booster (replaces Box2D). Pure Python floats: fast for one vehicle, no dependencies." and "The vehicle: a Falcon 9 first stage on its landing burn ... SI units".',
     "Our own simulator at real scale: first physics step, vehicle model (mass, thrust, gimbal, grid fins, RCS), unit tests for the physics."),
    ("2026-10-19 21:08", "src/sim/spawn.py (21:08), src/sim/calibrate.py (21:12)",
     '`class Wind` ("a per-episode mean plus a slow gust, stronger aloft (shear)"), re-entry start states ("arrives at SPAWN_HEIGHT_M (3 km) at up to ~300 m/s"), calibration of vehicle and fuel budget.',
     "Wind with shear in our own physics; difficulty levels L0-L3 re-defined with entry angles and speeds (src/utils/config.py)."),
    ("2026-10-19 22:50", "src/sim/eval_policy.py, dagger_loop.py (22:50); logs/_archive/bc_stage0_eroded ... bc3_improved_then_drifted (22:34-22:43), logs/custom_s0_bc4 (22:49)",
     '"DAgger rounds: python -m src.sim.dagger_loop"; archived behaviour-cloning runs named bc2_accuracy_over_speed, bc3_improved_then_drifted.',
     "Imitation experiments with logged outcomes (archived episodes.csv files, 349-2,099 episodes each)."),
    ("2026-10-20 05:27", "src/sim/eval_expert.py (05:10), src/sim/expert.py (05:27), tests",
     '"A scripted landing controller (\'expert\'): an engine-first descent, a throttle that tracks a braking profile, attitude control with the right actuator for the airflow (fins at high dynamic pressure, gimbal under thrust, RCS as the backup)".',
     "A scripted controller that lands reliably at level 0; used for demonstrations. Later tuned (17:27-17:31, tools/expert_tuning, the pre-tuning copy is kept as expert_before_tune.py)."),
    ("2026-10-20 09:13", "src/sim/dual_env.py (09:13), src/training/pair.py (10:07), src/sim/pipeline.py (09:35), tests/test_dual.py, test_pair.py, test_pipeline.py",
     '"DualEnv (\'Dual land\'): ONE environment with TWO landers and TWO pads", "Two neural networks in one environment", and the pipeline "expert demonstrations -> behaviour cloning -> DAgger rounds -> evaluation".',
     "Pad A/B dual landing; pair training run custom_s0_pair (1,646 episodes, ~1.0 M steps, 21 Oct; train_custom_s0_pair.png)."),
    ("2026-10-20 11:35", "src/sim/env.py (11:35), tests/test_sx_env.py (11:16-11:39), HANDOFF.md (13:21)",
     '"SpaceXEnv: the booster\'s landing burn as a Gymnasium environment on our own physics (no Box2D)." (18 observations, 5 actions). HANDOFF.md of 20 Oct: "123 tests pass".',
     "The single-lander environment on the new physics; imitation network sx_imitation_quiet: 30/30 level-0 landings in the hand-off note."),
    ("2026-10-20 15:49", "src/app/make_cam_hull.py (15:49), src/app/camhulls.py (21 Oct 06:36), src/app/cam_hulls/*.png, src/app/models/airflow_stage.png (21 Oct 07:23)",
     '"Draws ... the stage\'s hull as the landing cam sees it, computed (not drawn by hand) by ray-tracing the real stage geometry from the camera\'s real mounting."; hull pictures chosen per episode; airflow stage picture.',
     "The landing cam and partner cam with a ray-traced hull overlay (img_03)."),
]


def listing(root: Path, skip: set[str], title: str, out: list[str]):
    rows = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in skip and d != "__pycache__" and not d.startswith(".pytest")]
        for f in fn:
            p = Path(dp) / f
            if p.suffix.lower() in (".py", ".html", ".md", ".ps1", ".bat", ".ini", ".txt", ".png", ".gif"):
                st = p.stat()
                rows.append((st.st_ctime, st.st_mtime, p.relative_to(root).as_posix(), st.st_size))
    rows.sort()
    out += ["", title, "created (ctime)    modified (mtime)   bytes   file"]
    out += [f"{fmt(c)}   {fmt(m)}   {s:>8}  {p}" for c, m, p, s in rows]
    return rows


def main():
    out = ["FILE HISTORY OF SPACEXPERIMENT (there is NO git repository and no other copy of this project on this machine: dates are file-system times)",
           f"generated {fmt(time.time())} by paper_assets/progress/scripts/make_history.py",
           "Rocketeer folder = the earlier prototype (gymnasium Box2D LunarLander-v3), created 2026-10-18; SpaceXperiment folder = the rebuild on our own physics, created 2026-10-19 21:02."]
    a = listing(PROTO, {"logs", "models", "data", "results", "notebooks", "SpaceXperiment", ".pytest_cache"}, "A. ROCKETEER PROTOTYPE (C:\\Users\\firdo\\rocketeer, without SpaceXperiment/)", out)
    b = listing(SX, {"logs", "models", "data", "results", "notebooks", "paper_assets", ".pytest_cache"}, "B. SPACEXPERIMENT (C:\\Users\\firdo\\rocketeer\\SpaceXperiment, without paper_assets/)", out)
    out += ["", "C. TRAINING RUN FOLDERS (episodes.csv: rows = finished training episodes; file time = last write)"]
    for base in (PROTO / "logs", SX / "logs", SX / "logs" / "_archive"):
        for d in sorted(base.glob("*/")):
            e = d / "episodes.csv"
            if e.exists():
                n = sum(1 for _ in open(e, encoding="utf-8", errors="ignore")) - 1
                out.append(f"{fmt(e.stat().st_mtime)}  {n:>6} episodes  {d.relative_to(PROTO).as_posix()}")
    (OUT / "file_history.txt").write_text("\n".join(out) + "\n", encoding="utf-8")

    md = ["# Milestones (dates are file-system times, not commits: this project is not under git)", "",
          "Each row quotes the file's own docstring or code. Time is creation time for new files and the last modification for copied ones; both are in `file_history.txt`.", "",
          "| # | date | evidence (files, time) | what changed | what can be seen or measured |", "|---|---|---|---|---|"]
    for i, (d, ev, ch, see) in enumerate(MILESTONES, 1):
        md.append(f"| {i} | {d} | {ev} | {ch} | {see} |")
    (OUT / "milestones.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    first = min(r[0] for r in a + b)
    last = max(r[1] for r in a + b)
    print("first file", fmt(first), "last modification", fmt(last), "| files", len(a), len(b))


if __name__ == "__main__":
    main()
