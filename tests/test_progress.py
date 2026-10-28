import math

import numpy as np

from src.app import server
from src.training.env import make_env
from src.utils import config as C


def test_episode_metrics_carry_the_descent_quality_numbers():
    env = make_env(0, "custom", randomize=True, fuel_budget=1e9)
    rng = np.random.default_rng(0)
    seen = 0
    for ep in range(12):
        env.reset(seed=ep)
        for _ in range(2000):
            _, _, term, trunc, info = env.step(rng.uniform(-1, 1, 5).astype(np.float32))
            if term or trunc:
                m = info["episode_metrics"]
                assert (math.isnan(m["env_ok"]) or 0.0 <= m["env_ok"] <= 1.0) and 0.0 <= m["brake_peak"] <= 3.0
                seen += 1
                break
    assert seen == 12


def test_tail_bins_the_new_columns_and_tolerates_old_files(tmp_path):
    new = tmp_path / "new.csv"
    new.write_text("timesteps,success,crashed,out_of_fuel,soft,merit,speed_mps,dist_m,env_ok,brake_peak\n"
                   "100,False,True,False,False,10,6.0,3.0,0.5,1.5\n200,True,False,False,True,90,1.0,1.0,0.9,0.5\n300,False,True,False,False,0,nan,5.0,0.1,3.0\n")
    r = server.series(server.Tail(new).update())[0]
    assert abs(r["env"] - 0.5) < 1e-9 and abs(r["brake"] - (1.5 + 0.5 + 3.0) / 3) < 1e-9
    assert abs(r["td_speed"] - 3.5) < 1e-9 and abs(r["td_frac"] - 2 / 3) < 1e-9 and abs(r["dist"] - 3.0) < 1e-9
    old = tmp_path / "old.csv"
    old.write_text("timesteps,success,crashed,out_of_fuel,soft,merit\n100,False,True,False,False,10\n")
    r = server.series(server.Tail(old).update())[0]
    assert r["env"] is None and r["brake"] is None and r["td_speed"] is None


def test_altitude_band_overspeed_is_recorded_and_binned(tmp_path):
    env = make_env(0, "custom", randomize=True, fuel_budget=1e9)
    env.reset(seed=1)
    for _ in range(2000):
        _, _, term, trunc, info = env.step(np.zeros(5, np.float32))
        if term or trunc:
            break
    m = info["episode_metrics"]
    ov = [m[f"ov_{i}"] for i in range(len(C.ALT_BANDS_M) + 1)]
    assert any(v == v for v in ov) and all(v != v or 0.0 <= v <= 4.0 for v in ov)           # a free fall visited bands, each ratio in [0, 4]
    assert ov[0] == ov[0] and ov[0] >= 1.0                                                   # free fall is faster than the envelope near the ground (band below 30 m)
    header = "timesteps,success,crashed,out_of_fuel,soft,merit," + ",".join(f"ov_{i}" for i in range(8)) + "\n"
    f = tmp_path / "b.csv"
    f.write_text(header + "100,False,True,False,False,0,2.0,1.0,nan,nan,nan,nan,nan,nan\n200,False,True,False,False,0,4.0,3.0,nan,nan,nan,nan,nan,nan\n")
    r = server.series(server.Tail(f).update())[0]
    assert r["bands"][0] == 3.0 and r["bands"][1] == 2.0 and r["bands"][2] is None and len(r["bands"]) == 8
