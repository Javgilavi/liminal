import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import (  # noqa: E402
    DEFAULT_SPAN_TICKS,
    camera_role,
    canonical_joint,
    degrees_to_m100,
    guess_units,
    m100_to_degrees,
)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("left_shoulder_pan.pos", ("left", "shoulder_pan")),
        ("left_arm_wrist_roll.pos", ("left", "wrist_roll")),
        ("right_arm_gripper", ("right", "gripper")),
        ("right_gripper", ("right", "gripper")),
        ("main_elbow_flex", (None, "elbow_flex")),
        ("shoulder_lift.pos", (None, "shoulder_lift")),
        ("head_motor_1.pos", (None, None)),
        ("x.vel", (None, None)),
        ("gantry.height_mm", (None, None)),
    ],
)
def test_canonical_joint(name, expected):
    assert canonical_joint(name) == expected


@pytest.mark.parametrize(
    "key, role",
    [
        ("observation.images.head", "scene"),
        ("observation.images.left_top", "scene"),
        ("observation.images.right_front", "scene"),
        ("observation.images.left_wrist", "wrist:left"),
        ("observation.images.wrist_right", "wrist:right"),
        ("observation.images.left_left_wrist", "wrist:left"),
        ("observation.images.right_follower_right", "wrist:right"),
        ("observation.images.left", "unknown:left"),
        ("observation.images.wrist", "wrist:?"),
    ],
)
def test_camera_role(key, role):
    assert camera_role(key) == role


def test_guess_units():
    assert guess_units("wrist_roll", -150.0, 120.0, 0.0) == "degrees"
    assert guess_units("shoulder_lift", 51.7, 189.6, 120.0) == "degrees_legacy"
    assert guess_units("elbow_flex", -97.0, 98.0, 0.0) == "range_m100_100_or_degrees"
    assert guess_units("gripper", 0.0, 45.0, 10.0) == "range_0_100"
    assert guess_units("gripper", 0.0, 0.0, 0.0) == "constant"


def test_m100_degrees_roundtrip_and_full_turn():
    # wrist_roll spans the full 4095 ticks, so +-100 maps to +-180 degrees
    assert m100_to_degrees(100.0, "wrist_roll") == pytest.approx(180.0, abs=0.05)
    for j in DEFAULT_SPAN_TICKS:
        assert degrees_to_m100(m100_to_degrees(37.5, j), j) == pytest.approx(37.5)
