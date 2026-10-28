"""Rules-stage gate: decides when the newer environment rules switch on (pure logic, unit-tested)."""
from __future__ import annotations

from collections import deque


class RulesGate:
    """Tracks the share of survived landings over a window of episodes played at the CURRENT stage and advances the
    stage when it reaches `rate`, or after `max_steps` steps in the stage (0 = never)."""

    def __init__(self, stage: int, max_stage: int, window: int, rate: float, max_steps: int = 0, cap_min_rate: float = 0.0):
        self.stage, self.max_stage = int(stage), int(max_stage)
        self.window, self.rate, self.max_steps = int(window), float(rate), int(max_steps)
        self.cap_min_rate = float(cap_min_rate)       # the step cap only advances the stage if survival is at least this (0 = the cap always advances)
        self.results: deque = deque(maxlen=self.window)
        self.stage_start = 0

    def observe(self, episode_stage: int, survived: bool) -> None:
        """Record an episode. Episodes still running under an older stage are ignored."""
        if episode_stage == self.stage:
            self.results.append(1.0 if survived else 0.0)

    @property
    def survival(self) -> float:
        return sum(self.results) / len(self.results) if self.results else 0.0

    def check(self, num_timesteps: int, blocked: bool = False):
        """Returns (new_stage, reason, survival_before) if the stage advances now, else None. `blocked` holds the stage (cooldown)."""
        if self.stage >= self.max_stage or blocked:
            return None
        full = len(self.results) == self.window
        reason = None
        if full and self.survival >= self.rate:
            reason = "survival"
        elif self.max_steps > 0 and num_timesteps - self.stage_start >= self.max_steps and full and self.survival >= self.cap_min_rate:
            reason = "max_steps"                      # the cap is only a fallback for an agent that has really started to survive
        if reason is None:
            return None
        rate = self.survival
        self.stage += 1
        self.results.clear()
        self.stage_start = num_timesteps
        return self.stage, reason, rate


def level_reason(window_full: bool, rate: float, steps_at_level: int, advance_rate: float, max_steps: int, cap_min_rate: float):
    """Why the curriculum level should advance now: 'success' (rate over the window reached the gate), 'max_steps' (the cap, but only for an
    agent whose rate is at least cap_min_rate) or None (keep training at this level)."""
    if window_full and rate >= advance_rate:
        return "success"
    if steps_at_level >= max_steps and window_full and rate >= cap_min_rate:
        return "max_steps"
    return None
