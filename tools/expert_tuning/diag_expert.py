import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from src.sim import vehicle as V
from src.sim.expert import Expert
from src.sim.physics import step
from src.sim.spawn import Wind, make_spawn
from src.utils import config as C

def fly(level, seed):
    rng = np.random.default_rng(seed)
    w = Wind(C.LEVELS[level]["wind_power"], rng)
    s, d = make_spawn(rng, level, prop=C.FUEL_BUDGET, wind=w)
    ex = Expert(); ex.reset()
    cmd = (0, 0, 0, 0, 0); nxt = 0.0; t_burn = None; sat = 0; nb = 0; maxtilt = 0; h_burn = None; v_burn = None
    prev = (s.vx, s.vy)
    while s.t < 120:
        wx = w.at(s.t, s.y)
        if s.t >= nxt:
            cmd = ex.act(s, wx); nxt += 0.05
            if ex.burning and t_burn is None:
                t_burn = s.t; h_burn = s.foot_height(); v_burn = (s.vx, s.vy)
            if ex.burning:
                nb += 1; sat += cmd[0] >= 0.99
        prev = (s.vx, s.vy)
        step(s, *cmd, 0.01, wx)
        maxtilt = max(maxtilt, abs(s.th))
        if s.foot_height() <= 0: break
    return dict(d=d, speed=math.hypot(*prev), vy=prev[1], vx=prev[0], tilt=abs(s.th), x=s.x, sat=sat / max(nb, 1), h_burn=h_burn, v_burn=v_burn, maxtilt=maxtilt, prop=s.prop)

lv = int(sys.argv[1]); n = int(sys.argv[2])
bad = []
for i in range(n):
    r = fly(lv, 100 + 1000 * lv + i)
    crash = (-r["vy"] > 5) or abs(r["vx"]) > 3 or r["tilt"] > 0.2
    if crash: bad.append(r)
print("crashes", len(bad), "of", n)
for r in bad[:14]:
    print(f"entry {r['d']['entry_deg']:.0f}deg v0 {r['d']['speed']:.0f} offs {r['d']['target_miss_m']:.0f} | burn at h={r['h_burn']:.0f} v=({r['v_burn'][0]:.0f},{r['v_burn'][1]:.0f}) | td vy {r['vy']:.1f} vx {r['vx']:.1f} tilt {r['tilt']:.2f} | sat {r['sat']:.2f} maxtilt {r['maxtilt']:.2f} prop {r['prop']:.0f}")
