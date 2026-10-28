"""python src/training/train.py --reward custom|default --seed N [--stop-at-level L] [--total-steps N]"""
import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.training.brain import episode_payload
from src.training.curriculum import RulesGate, level_reason
from src.training.data import TrajectoryLogger
from src.training.env import make_vec_env
from src.utils import config as C

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

EP_FIELDS = ["timesteps", "level", "rules_stage", "success", "soft", "merit", "touchdown", "dist_m", "speed_mps", "tilt_rad",
             "fuel_used", "out_of_fuel", "end", "crashed", "tilt_out", "stall_out", "ignitions", "steps", "env_ok", "brake_peak", "sync_dt_s"] + [f"ov_{i}" for i in range(len(C.ALT_BANDS_M) + 1)]


def resolve_device() -> str:
    """The device for the network, from config DEVICE ('cpu', 'cuda' or 'auto'). 'cuda' fails loudly if this PyTorch
    build cannot use the GPU; 'auto' falls back to the CPU."""
    want = C.DEVICE
    if want == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if want == "auto":
        print("[device] no CUDA-enabled PyTorch found, training on the CPU", flush=True)
        return "cpu"
    sys.exit(f"Device 'cuda' was requested but this PyTorch build cannot use the GPU (torch {torch.__version__}, CUDA {torch.version.cuda}). "
             "Install a CUDA build of PyTorch, or set the device to 'cpu' or 'auto'.")


class CurriculumCallback(BaseCallback):
    """Logs every finished episode; advances the level when the last-200 success rate >= 80% at the
    current level or after 500k steps there. Stops training when the gate at stop_at_level passes."""

    def __init__(self, run_dir: Path, ckpt_prefix: Path, start_level=0, stop_at_level=None, snapshot_every=0, live=True,
                 live_dir=None, live_env=None):
        super().__init__()
        self.live_env = live_env   # sweep agents stream into a shared live dir under their own env index
        self.live, self.live_n = live, 0
        self.live_last = {}      # env index -> wall time of last streamed episode
        self.live_paths = []
        self.obs_buf, self.act_buf = {}, {}      # per env: what the policy saw / did this episode (for the Brain view)
        self.live_dir = Path(live_dir) if live_dir else run_dir / "live"
        self.snapshot_every = snapshot_every
        self.next_snap = 0
        self.snap_dir = Path(f"{ckpt_prefix}_snaps")
        self.run_dir, self.ckpt_prefix = run_dir, ckpt_prefix
        self.level, self.stop_at_level = start_level, stop_at_level
        self.window = deque(maxlen=C.ADVANCE_WINDOW)
        self.level_start = 0
        self.term_keys = None
        self.dlog = None
        self.gate = RulesGate(C.RULES_STAGE_START, C.RULES_STAGE_MAX, C.RULES_GATE_WINDOW, C.RULES_GATE_RATE, C.RULES_GATE_MAX_STEPS, C.RULES_GATE_CAP_MIN_RATE)
        self.last_gate_step = None            # step of the last level or stage advance: the cooldown keeps two harder things from starting together

    def _on_training_start(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ep_f = open(self.run_dir / "episodes.csv", "w", newline="")
        self.lv_f = open(self.run_dir / "levels.csv", "w", newline="")
        self.lv = csv.writer(self.lv_f)
        self.lv.writerow(["timesteps", "level", "reason", "success_rate"])
        self.lv.writerow([0, self.level, "start", ""])
        self.st_f = open(self.run_dir / "stages.csv", "w", newline="")
        self.st = csv.writer(self.st_f)
        self.st.writerow(["timesteps", "stage", "reason", "survival_rate"])
        self.st.writerow([0, self.gate.stage, "start", ""])
        self.st_f.flush()
        self.training_env.env_method("set_rules_stage", self.gate.stage)
        self.ep = None
        self.training_env.env_method("set_level", self.level)
        self._snapshot()
        if C.DATA_LOG and C.DATA_EVERY > 0:
            self.dlog = TrajectoryLogger(self.run_dir, self.run_dir.name)
        if self.live:
            import shutil
            if self.live_env is None:      # a sweep orchestrator clears the shared dir itself
                shutil.rmtree(self.live_dir, ignore_errors=True)
            self.live_dir.mkdir(parents=True, exist_ok=True)

    def _write_live(self, traj, env_idx, buf_idx=0):
        """Stream a real training episode for the viewer. At most one per env every 1.5 s of wall
        time (training finishes episodes far faster than anyone can watch them)."""
        import os
        import time
        if self.live_env is not None:
            env_idx = self.live_env
        now = time.time()
        if now - self.live_last.get(env_idx, 0.0) < 1.5:
            return
        self.live_last[env_idx] = now
        self.live_n += 1
        if C.LIVE_ACTIVATIONS and self.obs_buf.get(buf_idx):
            try:                               # computed only for episodes that are actually streamed
                traj["act"] = episode_payload(self.model.policy, np.array(self.obs_buf[buf_idx]), np.array(self.act_buf[buf_idx]))
            except Exception as e:             # the brain view is optional: never let it stop training
                print(f"[brain] activations skipped: {type(e).__name__}: {e}", flush=True)
        self._extra_payload(traj, buf_idx)
        traj.update(id=self.live_n, env=env_idx, timesteps=int(self.num_timesteps))
        try:                                   # streaming is best-effort: it must never kill training
            tmp = self.live_dir / f"tmp_e{env_idx}.json"   # per-env name: sweep agents share this directory
            tmp.write_text(json.dumps(traj, separators=(",", ":")))
            path = self.live_dir / f"ep_{self.live_n:08d}_e{env_idx}.json"
            os.replace(tmp, path)
            self.live_paths.append(path)
            while len(self.live_paths) > 80:
                self.live_paths.pop(0).unlink(missing_ok=True)
        except OSError:
            pass

    def _extra_payload(self, traj, buf_idx):
        """Hook: a two-network run adds the second network's activations here."""

    def _save_level(self, path):
        self.model.save(path)

    def _snapshot(self):
        if self.snapshot_every:
            self.snap_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(self.snap_dir / f"step_{self.num_timesteps:09d}")
            self.next_snap = self.num_timesteps + self.snapshot_every

    def _write_episode(self, m):
        if self.ep is None:
            terms = sorted(k for k in m if k.startswith("term_"))
            self.ep = csv.DictWriter(self.ep_f, EP_FIELDS + terms, extrasaction="ignore")
            self.ep.writeheader()
        self.ep.writerow({**m, "timesteps": self.num_timesteps})

    def _on_step(self) -> bool:
        if self.snapshot_every and self.num_timesteps >= self.next_snap:
            self._snapshot()
        if self.live and C.LIVE_ACTIVATIONS:
            acts = self.locals.get("clipped_actions", self.locals.get("actions"))
            for i, o in enumerate(np.asarray(self.model._last_obs)):   # _last_obs = what the policy saw for these actions
                self.obs_buf.setdefault(i, []).append(o.copy())
                self.act_buf.setdefault(i, []).append(np.array(acts[i], dtype=np.float32))
        if self.dlog is not None:                      # the step-level experience log (sampled, see data.py)
            acts = self.locals.get("clipped_actions", self.locals.get("actions"))
            rewards = self.locals.get("rewards")
            for i, o in enumerate(np.asarray(self.model._last_obs)):
                self.dlog.step(i, o, acts[i], rewards[i] if rewards is not None else 0.0)
        for env_idx, info in enumerate(self.locals["infos"]):
            if self.live and "trajectory" in info:
                self._write_live(info.pop("trajectory"), env_idx, env_idx)
            m = info.get("episode_metrics")
            if m is None:
                continue
            self.obs_buf[env_idx], self.act_buf[env_idx] = [], []
            if self.dlog is not None:
                self.dlog.end_episode(env_idx, m, self.num_timesteps)
            self._write_episode(m)
            self.gate.observe(m.get("rules_stage", C.RULES_STAGE_START), m["end"] == "rest")
            if m["level"] == self.level:
                self.window.append(float(m["soft"]))        # gate on soft landings, not the rare strict success
        cooling = self.last_gate_step is not None and self.num_timesteps - self.last_gate_step < C.GATE_COOLDOWN_STEPS
        adv = self.gate.check(self.num_timesteps, blocked=cooling)
        if adv:
            self.last_gate_step = self.num_timesteps                                       # a newer set of rules switches on (at each env's next reset)
            stage, why, before = adv
            self.st.writerow([self.num_timesteps, stage, why, before])
            self.st_f.flush()
            self.training_env.env_method("set_rules_stage", stage)
            print(f"[rules] t={self.num_timesteps} stage ->{stage} ({why}, survival {before:.2f})", flush=True)
        rate = float(np.mean(self.window)) if self.window else 0.0
        if self.n_calls % 200 == 0:
            self.logger.record("rules/stage", self.gate.stage)
            self.logger.record("rules/survival", self.gate.survival)
            self.logger.record("curriculum/level", self.level)
            self.logger.record("curriculum/success_rate", rate)
            self.ep_f.flush()
        reason = level_reason(len(self.window) == C.ADVANCE_WINDOW, rate, self.num_timesteps - self.level_start, C.ADVANCE_SUCCESS_RATE,
                              C.ADVANCE_MAX_STEPS, C.ADVANCE_CAP_MIN_RATE)
        if reason and cooling:
            reason = None                          # a stage just advanced: wait out the cooldown, then this level is re-checked
        if reason:
            self.last_gate_step = self.num_timesteps
            self._save_level(f"{self.ckpt_prefix}_level{self.level}")
            if self.level == self.stop_at_level:
                self.lv.writerow([self.num_timesteps, self.level, f"stop_{reason}", rate])
                print(f"[curriculum] stop at level {self.level} ({reason}), success {rate:.2f}", flush=True)
                return False
            if self.level < C.N_LEVELS - 1:
                self.lv.writerow([self.num_timesteps, self.level + 1, reason, rate])
                print(f"[curriculum] t={self.num_timesteps} level {self.level}->{self.level + 1} "
                      f"({reason}, success {rate:.2f})", flush=True)
                self.level += 1
                self.training_env.env_method("set_level", self.level)
                self.window.clear()
                self.level_start = self.num_timesteps
            else:
                self.level_start = self.num_timesteps  # final level: just keep training
        self.lv_f.flush()
        return True

    def _on_training_end(self):
        if self.dlog is not None:
            self.dlog.flush()
        self.ep_f.close()
        self.lv_f.close()
        self.st_f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["custom", "default"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=C.TOTAL_TIMESTEPS)
    ap.add_argument("--n-envs", type=int, default=C.N_ENVS)
    ap.add_argument("--start-level", type=int, default=0)
    ap.add_argument("--stop-at-level", type=int, default=None)
    ap.add_argument("--snapshot-every", type=int, default=0, help="save a policy snapshot every N steps (0=off)")
    ap.add_argument("--no-live", action="store_true", help="do not stream episodes to the live viewer")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lr", type=float, default=None, help="learning rate for this run (default: PPO_KWARGS)")
    ap.add_argument("--run-dir", default=None, help="log directory (default logs/<name>); used by sweeps")
    ap.add_argument("--live-dir", default=None, help="shared live-episode directory (sweeps)")
    ap.add_argument("--live-env", type=int, default=None, help="env index to stream as (sweeps)")
    ap.add_argument("--init-from", default=None, help="start from this checkpoint's weights (name under models/checkpoints or a .zip path)")
    args = ap.parse_args()

    torch.set_num_threads(1)
    name = f"{args.reward}_s{args.seed}{args.tag}"
    run_dir = Path(args.run_dir) if args.run_dir else C.LOG_DIR / name
    ppo_kwargs = dict(C.PPO_KWARGS, **({"learning_rate": args.lr} if args.lr else {}))
    C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    C.MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if C.START_PROFILE == "pair":                  # Dual land with one network per lander: two PPO models on one environment (src/training/pair.py)
        from src.training.pair import run_pair
        C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        C.MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return run_pair(args, name, run_dir, ppo_kwargs, CurriculumCallback, resolve_device())
    vec = make_vec_env(args.n_envs, args.seed, record_all=not args.no_live, level=args.start_level, reward=args.reward)
    tb = None
    try:
        import tensorboard  # noqa: F401
        tb = str(run_dir / "tb")
    except ImportError:
        pass
    device = resolve_device()
    print(f"[device] training the network on {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""), flush=True)
    arch = dict(pi=C.net_arch(), vf=C.net_arch())            # the bigger network (the old project used 64 x 64)
    model = PPO("MlpPolicy", vec, seed=args.seed, device=device, verbose=0, tensorboard_log=tb, policy_kwargs=dict(net_arch=arch), **ppo_kwargs)
    if args.init_from:                       # continue an existing agent: its weights, a fresh optimiser and step counter
        from src.training.weights_io import resolve_checkpoint
        src = resolve_checkpoint(args.init_from)
        try:
            model.set_parameters(str(src), exact_match=False)
        except (ValueError, RuntimeError) as e:
            sys.exit(f"Cannot start from {src.name}: its network does not fit this start profile ({C.START_PROFILE}: {C.DUAL_OBS_DIM if C.START_PROFILE == 'dual' else 18} inputs). "
                     f"A dual run needs a dual checkpoint: python -m src.sim.dual_init --from <single checkpoint> --out <name>.  ({type(e).__name__}: {str(e)[:120]})")
        print(f"[init] weights loaded from {src.name}", flush=True)
    (C.MODEL_CONFIG_DIR / f"{name}.json").write_text(json.dumps(
        dict(vars(args), ppo=ppo_kwargs, fuel_budget=C.FUEL_BUDGET, levels=C.LEVELS, overrides=C.OVERRIDES), indent=2))
    cb = CurriculumCallback(run_dir, C.CHECKPOINT_DIR / name, args.start_level, args.stop_at_level, args.snapshot_every, live=not args.no_live,
                           live_dir=args.live_dir, live_env=args.live_env)
    model.learn(args.total_steps, callback=cb, tb_log_name="ppo")
    model.save(C.CHECKPOINT_DIR / name)
    vec.close()
    print(f"done: {args.total_steps} steps requested, ran {model.num_timesteps}", flush=True)


if __name__ == "__main__":
    main()
