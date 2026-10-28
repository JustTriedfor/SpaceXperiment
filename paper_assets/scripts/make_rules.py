"""Writes paper_assets/success_rules.txt and paper_assets/wind_levels.txt from the live code (values are read from src/utils/config.py and src/sim, not typed in).

    python paper_assets/scripts/make_rules.py
"""
from __future__ import annotations

import csv
import inspect
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.sim import env as E
from src.sim import spawn as S
from src.utils import config as C

OUT = ROOT / "paper_assets"


def main():
    src = inspect.getsource(E.SpaceXEnv._metrics)
    ok_lines = src[src.index("ok = ("):src.index("fuel_left")].rstrip()
    settle = inspect.getsource(E.SpaceXEnv._settle_verdict)
    lines = [
        "SUCCESS RULES AS IMPLEMENTED (src/sim/env.py, SpaceXEnv._metrics; thresholds from src/utils/config.py)",
        "",
        "An episode counts as a SUCCESS when ALL FOUR hold:",
        f"  1. touchdown speed  <= {C.SUCCESS_MAX_SPEED_MPS} m/s   (SUCCESS_MAX_SPEED_MPS; the speed of the FIRST foot contact, the magnitude of the velocity vector just before it)",
        f"  2. tilt             <= {C.SUCCESS_MAX_TILT_RAD} rad   (SUCCESS_MAX_TILT_RAD; the tilt where the lander ended up after settling, about {C.SUCCESS_MAX_TILT_RAD * 180 / 3.141592653589793:.2f} deg)",
        f"  3. pad offset       <= {C.SUCCESS_MAX_DIST_M} m      (SUCCESS_MAX_DIST_M; horizontal distance of the lander's centre from the pad centre, where it ended up)",
        "  4. it never ran out of propellant, and the episode ended 'at rest' (end == 'rest'), i.e. it survived the touchdown and settled upright:",
        f"       - the first contact must not be a crash at once: vertical speed <= {C.SURVIVE_MAX_VSPEED_MPS} m/s, sideways speed <= {C.SURVIVE_MAX_HSPEED_MPS} m/s, tilt <= {C.CONTACT_TILT_MAX} rad (CONTACT_TILT_MAX);",
        f"       - then it has SETTLE_S = {C.SETTLE_S} s to settle (the main engine is cut at touchdown, TOUCHDOWN_ENGINE_CUT = {C.TOUCHDOWN_ENGINE_CUT}); it ends early as 'rest' when upright (<= {C.SUCCESS_MAX_TILT_RAD} rad)",
        f"         and nearly still (< {C.SETTLE_REST_SPEED} m/s, spin < {C.SETTLE_REST_SPIN} rad/s) for {C.SETTLE_HOLD_S} s; at the deadline it is 'rest' if tilt <= {C.SURVIVE_MAX_TILT_RAD} rad and speed <= {3 * C.SETTLE_REST_SPEED} m/s, else a crash;",
        f"       - tipping past {C.TIP_OVER_RAD} rad at any time is a crash.",
        "",
        "The exact code of the success test:",
        "    " + ok_lines.replace("\n", "\n    "),
        "",
        f"The paper's statement 'touchdown speed at most 3 m/s' matches the code: SUCCESS_MAX_SPEED_MPS = {C.SUCCESS_MAX_SPEED_MPS}.",
        "",
        "Other numbers that frame an episode:",
        f"  propellant loaded (FUEL_BUDGET)         {C.FUEL_BUDGET} kg;   fuel_remaining_kg in episodes.csv = FUEL_BUDGET - fuel used",
        f"  spawn height (single lander)            {C.SPAWN_HEIGHT_M} m above the pad, entry speed {C.SPACEX_V_MIN}-{C.SPACEX_V_MAX} m/s x the level's speed fraction {[l['speed'] for l in C.LEVELS]}",
        f"  tilt limit at any time (episode ends)   {C.TILT_LIMIT_DEG} deg;  stall rule: slower than {C.STALL_SPEED_MPS} m/s above {C.STALL_MIN_HEIGHT_M} m;  at most {C.MAX_IGNITIONS} ignitions",
        "",
        "failure_reason in episodes.csv (one label per failed episode, first match in this order; failure_flags lists every rule that was broken):",
        "  out_of_fuel    the propellant ran out",
        f"  too_fast       first-contact speed > {C.SUCCESS_MAX_SPEED_MPS} m/s",
        f"  tilted         tilt > {C.SUCCESS_MAX_TILT_RAD} rad",
        f"  off_pad        pad offset > {C.SUCCESS_MAX_DIST_M} m",
        "  crashed_other  anything else that did not end at rest with all three limits met: no touchdown at all (tilt-out, stall, out of bounds, time out), or a fall over / failed settle",
        "  none           success",
        "",
        "Settle verdict code (src/sim/env.py::SpaceXEnv._settle_verdict):",
        "    " + settle.rstrip().replace("\n", "\n    "),
    ]
    (OUT / "success_rules.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:16]))

    w = [
        "WIND LEVELS AS IMPLEMENTED (src/utils/config.py LEVELS, src/sim/spawn.py Wind)",
        "",
        "level  wind_power (m/s)  turbulence_power  entry angle range (deg)  speed fraction  x_offset (m)",
    ]
    for i, l in enumerate(C.LEVELS):
        w.append(f"  L{i}     {l['wind_power']:<16}  {l['turbulence_power']:<16}  {l['entry_deg']!s:<23}  {l['speed']:<14}  {l['x_offset']}")
    w += ["", "The wind is NOT a constant mean of 0 / 5 / 10 / 15 m/s. src/sim/spawn.py::Wind, per episode:",
          "    mean   = uniform(-1, 1) * 0.6 * wind_power        (a signed, random mean for the episode)",
          "    amp    = 0.4 * wind_power                           (gust amplitude, a sine with a random period of 6-14 s)",
          "    wind(t, h) = shear(h) * (mean + amp * sin(2*pi*t/period + phase)),   shear(h) = 0.4 + 0.6 * min(h / 1500 m, 1)",
          "so the wind_power values 0 / 5 / 10 / 15 are a SCALE: the episode mean is within +-0.6 * wind_power (up to 0 / 3 / 6 / 9 m/s), and the wind at a given moment",
          "can reach (0.6 + 0.4) * wind_power = 0 / 5 / 10 / 15 m/s at height. In episodes.csv: wind_mean_mps = this episode's signed mean (before the shear factor), wind_scale_mps = wind_power, wind_gust_amp_mps = amp.", ""]
    ep = ROOT / "paper_assets" / "episodes.csv"
    if ep.exists():
        rows = [r for r in csv.DictReader(open(ep)) if r["controller"] == "expert_tuned"]
        w.append("Realised in episodes.csv (expert_tuned rows, 1 row per seed):")
        w.append("level  n    mean |wind_mean| (m/s)  max |wind_mean| (m/s)  theory mean |.| = 0.3*scale")
        for lv in range(4):
            v = [abs(float(r["wind_mean_mps"])) for r in rows if int(r["level"]) == lv]
            if v:
                w.append(f"  L{lv}   {len(v):<4} {statistics.mean(v):<22.3f} {max(v):<21.3f} {0.3 * C.LEVELS[lv]['wind_power']:.3f}")
    (OUT / "wind_levels.txt").write_text("\n".join(w) + "\n", encoding="utf-8")
    print("\n".join(w))


if __name__ == "__main__":
    main()
