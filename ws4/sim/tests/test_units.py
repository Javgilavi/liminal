"""LeRobot calibrated units <-> MJCF radians (see units.py docstring for the formulas and sources)."""

import json

import numpy as np
import pytest

from ws4.sim.geometry import load_geometry, val
from ws4.sim.units import ArmUnits, StationUnits, half_range_from_calibration


def lerobot_normalize_degrees(raw, rmin, rmax):
    """Copy of lerobot 0.5.1 MotorsBus._normalize, DEGREES branch."""
    mid = (rmin + rmax) / 2
    return (raw - mid) * 360 / 4095


def lerobot_normalize_m100(raw, rmin, rmax):
    """Copy of lerobot 0.5.1 MotorsBus._normalize, RANGE_M100_100 branch (drive_mode 0)."""
    raw = min(max(raw, rmin), rmax)
    return (raw - rmin) / (rmax - rmin) * 200 - 100


def test_degrees_zero_is_mjcf_zero():
    u = ArmUnits.from_geometry("degrees")
    q = u.to_radians([0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(q[:5], 0.0)
    assert q[5] == pytest.approx(val(load_geometry()["gripper"]["closed_rad"]))


def test_degrees_match_physical_encoder_angle():
    """A joint moved N ticks from mid-range is N * 360/4096 deg physically; LeRobot reports N * 360/4095."""
    u = ArmUnits.from_geometry("degrees")
    rmin, rmax = 1000, 3100
    mid = (rmin + rmax) / 2
    for ticks in (-700, -1, 0, 333, 900):
        deg = lerobot_normalize_degrees(mid + ticks, rmin, rmax)
        q = u.to_radians([deg, 0, 0, 0, 0, 50])[0]
        assert q == pytest.approx(np.deg2rad(ticks * 360 / 4096), abs=1e-9)


def test_round_trip_all_modes():
    rng = np.random.default_rng(0)
    for mode in ("degrees", "range_m100_100", "radians"):
        u = ArmUnits.from_geometry(mode)
        for _ in range(20):
            v = np.concatenate([rng.uniform(-80, 80, 5), rng.uniform(0, 100, 1)])
            if mode == "radians":
                v = v / 60
            np.testing.assert_allclose(u.from_radians(u.to_radians(v)), v, atol=1e-9)


def test_range_m100_100_with_calibration(tmp_path):
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    calib = {j: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 900 + 10 * i, "range_max": 3100 - 5 * i}
             for i, j in enumerate(names)}
    path = tmp_path / "arm.json"
    path.write_text(json.dumps(calib))
    u = ArmUnits.from_geometry("range_m100_100", calibration=path)
    assert u.half_range_ticks == half_range_from_calibration(calib)
    c = calib["elbow_flex"]
    mid = (c["range_min"] + c["range_max"]) / 2
    for ticks in (-1000, -250, 0, 400, 1000):
        norm = lerobot_normalize_m100(mid + ticks, c["range_min"], c["range_max"])
        q = u.to_radians([0, 0, norm, 0, 0, 0])[2]
        assert q == pytest.approx(np.deg2rad(ticks * 360 / 4096), abs=1e-9)


def test_xlerobot_style_calibration_keys():
    calib = {f"left_arm_{j}": {"range_min": 1000, "range_max": 3000}
             for j in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]}
    assert half_range_from_calibration(calib)["wrist_flex"] == 1000.0


def test_gripper_mapping():
    g = load_geometry()["gripper"]
    u = ArmUnits.from_geometry("degrees")
    assert u.gripper_to_radians(0) == pytest.approx(val(g["closed_rad"]))
    assert u.gripper_to_radians(100) == pytest.approx(val(g["open_rad"]))
    assert u.gripper_from_radians(u.gripper_to_radians(37.0)) == pytest.approx(37.0)


def test_station_units_order():
    su = StationUnits("degrees")
    v = np.array([10, 0, 0, 0, 0, 0, -10, 0, 0, 0, 0, 0], dtype=float)
    q = su.to_radians(v)
    assert q[0] > 0 > q[6]
    np.testing.assert_allclose(su.from_radians(q), v, atol=1e-9)
