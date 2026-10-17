import asyncio
import time
from dataclasses import replace

import pytest

from liminal.perception.evidence import EvidenceRef, FactObservation, PredicateSpec, Truth
from liminal.perception.world_state import WorldState


class FakeClock:
    def __init__(self, value=10.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def observation(
    clock,
    value=Truth.TRUE,
    *,
    evidence_id="frame-1",
    frame_seq=1,
    attempt_id="try-1",
    valid_until=None,
):
    return FactObservation(
        predicate="block_in_box",
        value=value,
        evidence=(EvidenceRef(evidence_id, camera_id="overhead", frame_seq=frame_seq),),
        observer_id="replay_scene",
        observer_version="1",
        observed_utc="2026-10-17T10:00:00Z",
        observed_mono_s=clock(),
        run_id="run-1",
        attempt_id=attempt_id,
        object_bindings={"object": "red_1", "container": "box_1"},
        valid_until_mono_s=valid_until,
    )


def observed_world(clock, *, ttl=5.0):
    world = WorldState(monotonic=clock, wall_time=lambda: 1_797_000_000.0)
    world.register_predicate(PredicateSpec("block_in_box", ttl))
    capability = world.register_observer("replay_scene", "1", {"block_in_box"})
    return world, capability


def test_absent_and_negated_absent_are_unknown():
    world = WorldState()
    assert world.evaluate("block_in_box") is Truth.UNKNOWN
    assert world.evaluate("!block_in_box") is Truth.UNKNOWN
    assert not world.holds("!block_in_box")


def test_registered_observation_has_bound_evidence_and_strict_writer_authority():
    clock = FakeClock()
    world, capability = observed_world(clock)
    with pytest.raises(PermissionError):
        world.set("block_in_box", True, source="planner")

    fact = observation(clock)
    record = world.observe(fact, capability)
    bindings = {"object": "red_1", "container": "box_1"}
    assert record.sequence == world.sequence == 1
    assert record.evidence_ids == ("frame-1",)
    assert world.evaluate("block_in_box", bindings=bindings) is Truth.TRUE
    assert world.evaluate("!block_in_box", bindings=bindings) is Truth.FALSE
    assert world.evaluate("block_in_box", bindings={"object": "blue_1", "container": "box_1"}) is Truth.UNKNOWN


def test_same_value_observation_refreshes_evidence_and_expiry():
    clock = FakeClock()
    world, capability = observed_world(clock, ttl=2.0)
    world.observe(observation(clock, evidence_id="frame-1"), capability)
    clock.advance(1.0)
    world.observe(observation(clock, evidence_id="frame-2", frame_seq=2), capability)
    records = world.evidence_records("block_in_box", {"object": "red_1", "container": "box_1"})
    assert len(records) == 2
    assert records[-1].sequence == 2
    clock.advance(1.1)
    assert world.evaluate("block_in_box", bindings={"object": "red_1", "container": "box_1"}) is Truth.TRUE
    clock.advance(1.0)
    assert world.evaluate("block_in_box", bindings={"object": "red_1", "container": "box_1"}) is Truth.UNKNOWN


def test_expiry_wakes_wait_until_violated():
    async def scenario():
        world = WorldState()
        world.register_predicate(PredicateSpec("lid_open", 0.03))
        capability = world.register_observer("camera", "v1", {"lid_open"})
        obs = FactObservation(
            predicate="lid_open",
            value=Truth.TRUE,
            evidence=(EvidenceRef("lid-frame"),),
            observer_id="camera",
            observer_version="v1",
            observed_utc="now",
            observed_mono_s=asyncio.get_running_loop().time(),
            run_id="run",
            attempt_id="attempt",
        )
        world.observe(obs, capability)
        await asyncio.wait_for(world.wait_until_violated("lid_open"), timeout=0.3)
        return world.evaluate("lid_open")

    assert asyncio.run(scenario()) is Truth.UNKNOWN


def test_bound_wait_expires_for_the_same_object_binding():
    async def scenario():
        world = WorldState()
        world.register_predicate(PredicateSpec("block_in_box", 0.02))
        capability = world.register_observer("replay_scene", "1", {"block_in_box"})
        world.observe(observation(time.monotonic), capability)
        bindings = {"object": "red_1", "container": "box_1"}
        await asyncio.wait_for(world.wait_until_violated("block_in_box", bindings=bindings), timeout=0.2)
        return world.evaluate("block_in_box", bindings=bindings)

    assert asyncio.run(scenario()) is Truth.UNKNOWN


def test_legacy_mock_set_remains_trusted_for_unregistered_predicates():
    world = WorldState()
    world.set("mock_effect", True, source="mock")
    assert world.holds("mock_effect")


def test_run_scope_hides_historical_live_evidence():
    clock = FakeClock()
    world, capability = observed_world(clock)
    world.observe(observation(clock), capability)
    bindings = {"object": "red_1", "container": "box_1"}
    world.begin_run("run-1")
    assert world.evaluate("block_in_box", bindings=bindings) is Truth.TRUE
    world.begin_run("run-2")
    assert world.evaluate("block_in_box", bindings=bindings) is Truth.UNKNOWN


def test_expired_latest_capture_does_not_resurrect_older_truth():
    clock = FakeClock()
    world, capability = observed_world(clock, ttl=10.0)
    world.observe(observation(clock, evidence_id="frame-1", frame_seq=1, valid_until=18.0), capability)
    clock.advance(1.0)
    world.observe(
        observation(clock, Truth.FALSE, evidence_id="frame-2", frame_seq=2, valid_until=14.0),
        capability,
    )
    clock.advance(3.1)
    bindings = {"object": "red_1", "container": "box_1"}
    assert world.evaluate("block_in_box", bindings=bindings) is Truth.UNKNOWN


def test_completed_attempt_and_regressed_frame_are_rejected():
    clock = FakeClock()
    world, capability = observed_world(clock)
    world.begin_run("run-1")
    world.begin_attempt("attempt-1")
    world.observe(observation(clock, attempt_id="attempt-1"), capability)
    world.end_attempt()
    with pytest.raises(PermissionError, match="completed attempt"):
        world.observe(observation(clock, evidence_id="late-frame", attempt_id="attempt-1"), capability)


def test_future_observation_and_frame_capture_are_rejected():
    clock = FakeClock()
    world, capability = observed_world(clock)
    future = observation(lambda: clock() + 1.0)
    with pytest.raises(ValueError, match="future"):
        world.observe(future, capability)
    obs = observation(clock)
    future_frame = FactObservation(
        predicate=obs.predicate,
        value=obs.value,
        evidence=(EvidenceRef("future", camera_id="overhead", frame_seq=2, capture_mono_s=clock() + 1),),
        observer_id=obs.observer_id,
        observer_version=obs.observer_version,
        observed_utc=obs.observed_utc,
        observed_mono_s=obs.observed_mono_s,
        run_id=obs.run_id,
        attempt_id=obs.attempt_id,
        object_bindings=obs.object_bindings,
    )
    with pytest.raises(ValueError, match="capture timestamp is in the future"):
        world.observe(future_frame, capability)


def test_frame_sequence_regression_is_rejected():
    clock = FakeClock()
    world, capability = observed_world(clock)
    world.observe(observation(clock, evidence_id="frame-3", frame_seq=3), capability)
    clock.advance(0.1)
    with pytest.raises(ValueError, match="frame sequence regressed"):
        world.observe(observation(clock, evidence_id="frame-2", frame_seq=2), capability)


def test_conflicting_same_time_observers_evaluate_unknown():
    clock = FakeClock()
    world, first_capability = observed_world(clock)
    second_capability = world.register_observer("second_detector", "2", {"block_in_box"})
    first = observation(clock)
    world.observe(first, first_capability)
    second = replace(
        first,
        value=Truth.FALSE,
        evidence=(EvidenceRef("second-frame", camera_id="wrist", frame_seq=1),),
        observer_id="second_detector",
        observer_version="2",
    )
    world.observe(second, second_capability)
    assert world.evaluate("block_in_box", bindings=first.object_bindings) is Truth.UNKNOWN


def test_late_older_observation_does_not_overwrite_newer_capture():
    clock = FakeClock()
    world, first_capability = observed_world(clock)
    second_capability = world.register_observer("second_detector", "2", {"block_in_box"})
    newer = observation(clock, value=Truth.TRUE)
    world.observe(newer, first_capability)
    older = replace(
        newer,
        value=Truth.FALSE,
        observed_mono_s=clock() - 0.5,
        evidence=(EvidenceRef("old-frame", camera_id="wrist", frame_seq=1),),
        observer_id="second_detector",
        observer_version="2",
    )
    world.observe(older, second_capability)
    assert world.evaluate("block_in_box", bindings=newer.object_bindings) is Truth.TRUE


def test_fake_clock_advance_wakes_bound_expiry_monitor():
    async def scenario():
        clock = FakeClock()
        world, capability = observed_world(clock, ttl=5.0)
        world.observe(observation(clock), capability)
        bindings = {"object": "red_1", "container": "box_1"}
        waiter = asyncio.create_task(world.wait_until_violated("block_in_box", bindings=bindings))
        await asyncio.sleep(0)
        clock.advance(5.0)
        world.notify_clock_advanced()
        await asyncio.wait_for(waiter, timeout=0.2)
        return world.evaluate("block_in_box", bindings=bindings)

    assert asyncio.run(scenario()) is Truth.UNKNOWN
