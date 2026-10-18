"""No hardware, time waits, or optional dependencies."""

import pytest

from liminal.safety.stall import StallMonitor


def sample(monitor, now, *, position=0.0, target=1.0, load=0.0):
    monitor.observe(now, {"wrist": position}, {"wrist": target}, {"wrist": load})


def test_position_error_alone_never_triggers_a_stall():
    monitor = StallMonitor()
    for now in (0, 0.5, 1, 10, 100):
        sample(monitor, now, load=0.1)


def test_frozen_joint_with_sustained_load_triggers_named_stall():
    monitor = StallMonitor()
    sample(monitor, 0, load=0.5)
    sample(monitor, 0.9, load=0.5)
    with pytest.raises(RuntimeError, match="possible stall on wrist"):
        sample(monitor, 1.01, load=0.5)


def test_real_progress_prevents_false_stall_even_with_high_following_error():
    monitor = StallMonitor()
    for i in range(100):
        q = i * 0.125
        sample(monitor, i * 0.2, position=q, target=q + 1, load=0.5)


def test_reduced_load_or_error_resets_the_sustained_condition():
    for reset in ({"load": 0.1}, {"target": 0.2}):
        monitor = StallMonitor()
        sample(monitor, 0, load=0.5)
        sample(monitor, 0.9, **reset)
        sample(monitor, 1.0, load=0.5)
        sample(monitor, 1.9, load=0.5)


def test_planning_pause_is_not_a_stall_with_low_load():
    monitor = StallMonitor()
    sample(monitor, 0, load=0.0)
    sample(monitor, 2, load=0.0)
    sample(monitor, 2.05, position=0.2, target=1.2, load=0.0)


def test_reset_discards_old_contact_evidence():
    monitor = StallMonitor()
    sample(monitor, 0, load=0.5)
    monitor.reset()
    sample(monitor, 10, load=0.5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 2.0])
def test_invalid_load_is_not_interpreted_as_no_load(bad):
    with pytest.raises(RuntimeError, match="invalid stall telemetry"):
        sample(StallMonitor(), 0, load=bad)
