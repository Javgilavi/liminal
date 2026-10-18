import asyncio
import time

from liminal.perception import (
    FixtureVisionModel,
    InspectionRequest,
    SceneAssessment,
    SceneInspector,
    Verdict,
    WorldState,
)


def test_scene_assessment_parses_valid_output_and_fails_closed():
    request = InspectionRequest(
        predicates=("lid_open", "access_clear", "access_blocked"),
        cameras=("top", "left_wrist"),
        exclusive_groups=(("access_clear", "access_blocked"),),
    )
    assessment = SceneAssessment.from_model_output(request, {
        "predicates": {
            "lid_open": {"value": "true", "evidence": "interior visible", "confidence": 0.9},
            "access_clear": {"value": False, "evidence": "cable crosses target"},
            "access_blocked": {"value": "true", "evidence": "cable crosses target"},
        }
    })
    assert assessment.observation("lid_open").value is Verdict.TRUE
    assert assessment.observation("access_clear").value is Verdict.FALSE
    assert assessment.observation("access_blocked").value is Verdict.TRUE

    malformed = SceneAssessment.from_model_output(request, "not JSON")
    assert all(not observation.known for observation in malformed.observations.values())

    contradictory = SceneAssessment.from_model_output(request, {
        "predicates": {
            "lid_open": {"value": "true"},
            "access_clear": {"value": "true"},
            "access_blocked": {"value": "true"},
        }
    })
    assert contradictory.observation("access_clear").value is Verdict.UNKNOWN
    assert contradictory.observation("access_blocked").value is Verdict.UNKNOWN


def test_world_state_requires_fresh_known_observations():
    world = WorldState()
    assert not world.holds("cable_clear") and not world.holds("!cable_clear")

    world.observe("cable_clear", Verdict.TRUE, source="vlm", evidence="cable is held aside")
    baseline = world.observation_version("cable_clear")
    assert world.holds("cable_clear")
    assert not world.holds("cable_clear", after_version=baseline)

    async def wait_for_reobservation():
        waiter = asyncio.create_task(world.wait_for("cable_clear", timeout=0.1, after_version=baseline))
        await asyncio.sleep(0)
        world.observe("cable_clear", True, source="vlm", evidence="confirmed again")
        return await waiter

    assert asyncio.run(wait_for_reobservation())
    assert world.observation_version("cable_clear") > baseline

    world.observe("cable_clear", Verdict.UNKNOWN, source="vlm")
    assert not world.holds("cable_clear") and not world.holds("!cable_clear")

    world.observe("lid_open", True, source="vlm", captured_at=time.monotonic() - 2.0)
    assert not world.holds("lid_open", max_age_s=0.5)


def test_scene_inspector_requires_temporal_agreement():
    response = {
        "predicates": {
            "lid_open": {"value": "true", "evidence": "box interior is visible"},
        }
    }
    model = FixtureVisionModel([response, response, {
        "predicates": {"lid_open": {"value": "false", "evidence": "lid is down"}}
    }])
    world = WorldState()

    class Camera:
        def read(self):
            return "frame"

    inspector = SceneInspector(model, {"top": Camera()}, world, confirmations=2)
    request = InspectionRequest(predicates=("lid_open",), context="Inspect the box lid.")

    async def scenario():
        first = await inspector.inspect(request)
        second = await inspector.inspect(request)
        conflict = await inspector.inspect(request)
        return first, second, conflict

    first, second, conflict = asyncio.run(scenario())
    assert first.observation("lid_open").value is Verdict.UNKNOWN
    assert second.observation("lid_open").value is Verdict.TRUE
    assert conflict.observation("lid_open").value is Verdict.UNKNOWN
    assert not world.holds("lid_open") and not world.holds("!lid_open")
    assert len(model.calls) == 3
    assert "not a planner" in model.calls[0][0]
