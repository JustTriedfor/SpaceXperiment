import math

import pytest

from src.sim import vehicle as V
from src.sim.physics import State, passive_descent, step
from src.utils import config as C


def run(s, seconds, dt=0.01, **cmd):
    for _ in range(int(seconds / dt)):
        step(s, cmd.get("throttle", 0.0), cmd.get("gimbal", 0.0), cmd.get("fin", 0.0), cmd.get("rcs_a", 0.0), cmd.get("rcs_b", 0.0), dt, cmd.get("wind", 0.0))
    return s


def test_vehicle_numbers_are_the_documented_ones():
    assert V.DRY_MASS == 22_200 and C.GRAVITY_MPS2 == 9.81
    assert V.MDOT_FULL == pytest.approx(914_000 / (311 * 9.80665), rel=1e-9) and 295 < V.MDOT_FULL < 305
    assert V.thrust_full(0.0) == pytest.approx(845_000) and V.thrust_full(80_000.0) == pytest.approx(914_000, rel=1e-3)
    s = State(prop=4500.0)
    twr_full = V.thrust_full(0.0) / (s.mass * C.GRAVITY_MPS2)
    twr_min = V.THROTTLE_MIN * twr_full
    assert 3.0 < twr_full < 3.6 and 1.2 < twr_min < 1.5            # cannot hover: even the minimum throttle out-pushes the weight
    assert 150 < V.terminal_velocity(s.mass, 0.0) < 230


def test_free_fall_in_near_vacuum_is_g():
    s = State(x=0, y=100_000.0, vx=30.0, vy=0.0, prop=4500.0)
    run(s, 5.0)
    assert s.vy == pytest.approx(-9.81 * 5, abs=0.15) and s.vx == pytest.approx(30.0, abs=0.05)
    assert s.y == pytest.approx(100_000 - 0.5 * 9.81 * 25, abs=1.0)


def test_engine_first_fall_reaches_about_terminal_velocity():
    s = State(x=0, y=3000.0, vy=-200.0, prop=4500.0)
    s.th = 0.0
    while s.foot_height() > 0 and s.t < 60:
        step(s, 0, 0, 0, 0, 0, 0.02)
    vt = V.terminal_velocity(s.mass, 0.0)
    assert 0.8 * vt < math.hypot(s.vx, s.vy) < 1.25 * vt          # the density rises on the way down, so it settles near the sea-level terminal speed


def test_thrust_acceleration_and_propellant_flow():
    s = State(x=0, y=1000.0, prop=4500.0)
    m0 = s.mass
    run(s, 10.0, throttle=1.0)
    assert s.ign == 1 and s.on
    burned = 4500.0 - s.prop
    assert burned == pytest.approx(V.MDOT_FULL * (10.0 - V.SPOOL_TAU), rel=0.03)      # about 300 kg/s minus the spool-up lag
    s2 = State(x=0, y=1000.0, prop=4500.0)
    run(s2, 3.0, throttle=1.0)
    a_up = (s2.vy - 0.0) / 3.0                                       # average acceleration (includes the spool-up)
    assert a_up > 15.0 and s2.vy > 0


def test_throttle_has_a_floor_and_an_off_state():
    s = State(x=0, y=5000.0, prop=4500.0)
    run(s, 3.0, throttle=0.05)
    assert s.thr == pytest.approx(V.THROTTLE_MIN, abs=0.02)         # asking for 5% gives the 40% floor
    run(s, 2.0, throttle=0.0)
    assert s.thr < 0.001 and not s.on


def test_the_engine_stops_when_the_propellant_is_gone():
    s = State(x=0, y=20_000.0, prop=100.0)
    run(s, 2.0, throttle=1.0)
    assert s.prop == 0.0 and s.out_of_prop and s.thr == 0.0 and s.thrust_n == 0.0


def test_actuator_signs():
    base = dict(x=0, y=20_000.0, prop=4500.0)
    s = run(State(**base), 2.0, throttle=1.0, gimbal=V.GIMBAL_MAX)
    assert s.om > 0                                                  # positive gimbal -> counter-clockwise torque
    a = run(State(**base), 2.0, rcs_a=1.0)
    b = run(State(**base), 2.0, rcs_b=1.0)
    assert a.om < 0 < b.om                                           # an upper push to the right turns it clockwise, a lower one counter-clockwise
    assert a.vx > 0 and b.vx > 0                                     # both push the vehicle to the right
    both = run(State(**base), 2.0, rcs_a=1.0, rcs_b=1.0)
    assert abs(both.om) < abs(a.om) and both.vx > a.vx               # together: translation with little rotation
    f = State(x=0, y=1000.0, vy=-250.0, prop=4500.0)
    run(f, 0.4, fin=V.FIN_MAX)
    assert f.om < 0 and f.vx > 0                                     # a positive fin deflection in the airstream pushes right, torque clockwise


def test_fins_need_airflow_and_rcs_runs_out_of_gas():
    still = State(x=0, y=20_000.0, prop=4500.0)
    run(still, 2.0, fin=V.FIN_MAX)
    assert abs(still.om) < 1e-3                                       # no dynamic pressure, no authority
    s = State(x=0, y=100_000.0, prop=4500.0)
    run(s, 40.0, rcs_a=1.0, rcs_b=1.0)
    assert s.gas == 0.0 and s.cmd_a == 0.0                            # empty: the commands are ignored
    assert V.RCS_GAS - s.gas == pytest.approx(V.RCS_GAS)


def test_engine_first_flight_is_aerodynamically_stable_and_damped():
    s = State(x=0, y=3000.0, vy=-250.0, th=0.3, prop=4500.0)         # 17 degrees off the airstream
    first, second = 0.0, 0.0
    while s.t < 12.0:
        step(s, 0, 0, 0, 0, 0, 0.01)
        if s.t < 6.0:
            first = max(first, abs(s.th))
        else:
            second = max(second, abs(s.th))
    assert first <= 0.31 and second < 0.8 * first                     # a weathercock: swings back through the airstream and the swings shrink (slowly: only the body damps it)


def test_feet_geometry():
    s = State(x=0, y=100.0)
    (xa, ya), (xb, yb) = s.feet()
    assert ya == pytest.approx(100 - C.LEG_DOWN_M) and yb == pytest.approx(ya) and xb - xa == pytest.approx(V.LEG_SPAN)
    s.th = 0.1
    assert s.foot_height() < 100 - C.LEG_DOWN_M + 1e-9 and abs(s.feet()[0][1] - s.feet()[1][1]) > 1.0   # tilted: one foot lower


def test_passive_descent_predicts_the_impact():
    s = State(x=-900.0, y=3000.0, vx=150.0, vy=-260.0, th=math.atan2(150.0, 260.0), prop=4500.0)
    x, v, t = passive_descent(s)
    assert 200 < v < 300 and 8 < t < 16
    assert -900 < x < 1200 and s.t == 0.0 and s.x == -900.0          # the input is untouched
