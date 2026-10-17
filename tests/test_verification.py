import asyncio
import time

from liminal.perception.evidence import EvidenceRef, FactObservation, PredicateSpec, Truth
from liminal.perception.verifier import VerificationRequest, Verifier
from liminal.perception.world_state import WorldState


def setup_world(ttl=2.0):
    world = WorldState()
    world.register_predicate(PredicateSpec("block_in_box", ttl))
    good = world.register_observer("detector", "v2", {"block_in_box"})
    other = world.register_observer("other", "v1", {"block_in_box"})
    return world, good, other


def make_observation(value, *, run="run-7", attempt="attempt-3", observer="detector", evidence_id="frame"):
    return FactObservation(
        predicate="block_in_box",
        value=value,
        evidence=(EvidenceRef(evidence_id, camera_id="overhead", frame_seq=12),),
        observer_id=observer,
        observer_version="v2" if observer == "detector" else "v1",
        observed_utc="2026-10-17T10:00:00Z",
        observed_mono_s=time.monotonic(),
        run_id=run,
        attempt_id=attempt,
        object_bindings={"object": "red_1", "container": "box_1"},
    )


def request(watermark, *, deadline=None, evidence_after_mono=None):
    return VerificationRequest(
        run_id="run-7",
        attempt_id="attempt-3",
        expected_effect="block_in_box",
        object_bindings={"object": "red_1", "container": "box_1"},
        evidence_after_seq=watermark,
        allowed_observers=("detector",),
        deadline_mono_s=deadline if deadline is not None else time.monotonic() + 0.2,
        evidence_after_mono_s=evidence_after_mono,
    )


def test_verifier_requires_post_watermark_attempt_specific_evidence():
    world, good, _ = setup_world()
    world.observe(make_observation(Truth.TRUE, attempt="previous"), good)
    watermark = world.sequence
    world.observe(make_observation(Truth.TRUE), good)
    result = asyncio.run(Verifier().verify(request(watermark), world))
    assert result.decision is Truth.TRUE
    assert result.evidence_ids == ("frame",)
    assert result.evidence_sequence == watermark + 1


def test_verifier_rejects_stale_wrong_run_attempt_observer_and_binding():
    world, good, other = setup_world()
    watermark = world.sequence
    world.observe(make_observation(Truth.TRUE, run="old-run"), good)
    world.observe(make_observation(Truth.TRUE, attempt="old-attempt"), good)
    world.observe(make_observation(Truth.TRUE, observer="other"), other)
    world.observe(
        FactObservation(
            predicate="block_in_box",
            value=Truth.TRUE,
            evidence=(EvidenceRef("other-object"),),
            observer_id="detector",
            observer_version="v2",
            observed_utc="now",
            observed_mono_s=time.monotonic(),
            run_id="run-7",
            attempt_id="attempt-3",
            object_bindings={"object": "blue_1", "container": "box_1"},
        ),
        good,
    )
    result = asyncio.run(Verifier().verify(request(watermark, deadline=time.monotonic()), world))
    assert result.decision is Truth.UNKNOWN
    assert result.missing_evidence == ("block_in_box",)


def test_verifier_returns_fresh_negative_observation():
    world, good, _ = setup_world()
    watermark = world.sequence
    world.observe(make_observation(Truth.FALSE), good)
    result = asyncio.run(Verifier().verify(request(watermark), world))
    assert result.decision is Truth.FALSE
    assert result.evidence_ids == ("frame",)


def test_verifier_waits_for_new_evidence_until_deadline():
    async def scenario():
        world, good, _ = setup_world()
        watermark = world.sequence
        req = request(watermark, deadline=time.monotonic() + 0.3)

        async def publish():
            await asyncio.sleep(0.02)
            world.observe(make_observation(Truth.TRUE), good)

        task = asyncio.create_task(publish())
        result = await Verifier().verify(req, world)
        await task
        return result

    assert asyncio.run(scenario()).decision is Truth.TRUE


def test_verifier_rejects_late_acceptance_of_a_predating_capture():
    async def scenario():
        world, good, _ = setup_world()
        watermark = world.sequence
        old_frame = make_observation(Truth.TRUE)
        capture_boundary = time.monotonic()
        request_value = request(
            watermark,
            deadline=capture_boundary + 0.2,
            evidence_after_mono=capture_boundary,
        )
        await asyncio.sleep(0.005)
        world.observe(old_frame, good)
        return await Verifier().verify(request_value, world)

    result = asyncio.run(scenario())
    assert result.decision is Truth.UNKNOWN
    assert result.reason == "no matching fresh evidence before deadline"


def test_verifier_rejects_an_already_expired_request_even_with_newer_evidence():
    world, good, _ = setup_world()
    watermark = world.sequence
    world.observe(make_observation(Truth.TRUE), good)
    expired = request(watermark, deadline=time.monotonic() - 0.01)
    result = asyncio.run(Verifier().verify(expired, world))
    assert result.decision is Truth.UNKNOWN
    assert result.reason == "verification deadline expired"
