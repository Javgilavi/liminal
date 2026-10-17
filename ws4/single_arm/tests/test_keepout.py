"""FK keep-out: replays of the two A0 attempts that preceded the elbow damage, and of clean hand-guided demos."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
FIX = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from keepout import KeepOut  # noqa: E402

REST = np.array([-13.8, -102.0, 96.7, 38.0, -99.2, 0.4])  # measured folded rest pose of red_follower


def first_refusal(ko, commands, states):
    for i, (c, s) in enumerate(zip(commands, states, strict=True)):
        refuse, zone = ko.refuse(c, s)
        if refuse:
            return i, zone
    return None, ""


def load_climb_tick(name, threshold=500):
    loads = json.loads((FIX / f"{name}_health.json").read_text())["elbow_load_per_s"]
    return 30 * next(i for i, v in enumerate(loads) if v is not None and abs(v) > threshold)


@pytest.mark.parametrize("name", ["a0_attempt2", "a0_attempt3"])
def test_damaging_attempts_are_refused_before_the_elbow_load_climbs(name):
    trace = np.load(FIX / f"{name}_trace.npz")
    tick, zone = first_refusal(KeepOut(), trace["command"], trace["state"])
    assert tick is not None and tick < load_climb_tick(name) - 90, (tick, zone)  # at least 3 s of margin


@pytest.mark.parametrize("name", ["a0_attempt2", "a0_attempt3"])
def test_base_and_fold_zones_catch_the_fold_back_on_their_own(name):
    # Same replays with the table check disabled: the base / fold-back zones alone still refuse in time.
    ko = replace(KeepOut(), table_z=-1.0)
    trace = np.load(FIX / f"{name}_trace.npz")
    tick, zone = first_refusal(ko, trace["command"], trace["state"])
    assert zone in ("base", "fold") and tick < load_climb_tick(name), (tick, zone)


@pytest.mark.parametrize("name", ["handguided_ep2", "handguided_ep3"])
def test_clean_cup_stacking_demos_are_never_refused(name):
    q = np.load(FIX / f"{name}.npz")["q"]
    ko = KeepOut()
    assert not any(ko.refuse(q[i + 1], q[i])[0] for i in range(len(q) - 1))


def test_rest_pose_is_clear_and_leaving_a_zone_is_allowed():
    ko = KeepOut()
    assert np.all(ko.depths(REST) == 0)
    inside = REST.copy()
    inside[1], inside[2] = -40.0, 90.0  # TCP down near the table in front of the base
    assert ko.depths(inside).sum() > 0
    out = inside.copy()
    out[1] = -60.0  # back up toward rest: shallower, so allowed even though still in a zone
    assert ko.depths(out).sum() < ko.depths(inside).sum()
    assert not ko.refuse(out, inside)[0]
    assert ko.refuse(inside, out)[0]


def test_guard_holds_the_measured_pose_when_refusing():
    import run_single_arm as rsa

    measured = np.concatenate([REST, rsa.PHANTOM_STATE])
    guard = rsa.KeepOutGuard(rsa.SafetyConfig(lower=-np.full(12, 1e3), upper=np.full(12, 1e3),
                                              max_step=np.full(12, 1e3)), KeepOut(), "left",
                             lambda: REST)  # fmt: skip
    bad = measured.copy()
    bad[1], bad[2] = -40.0, 90.0
    out, report = guard.filter(bad, measured)
    assert np.allclose(out[:5], REST[:5]) and report.intervened and guard.counts[guard.last_zone] == 1
