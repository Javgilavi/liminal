"""Fixture-driven end-to-end coverage for the component-replacement tree."""

import asyncio

from liminal.core.types import Status
from liminal.orchestration import load_task, run_task
from liminal.perception import FixtureVisionModel, SceneAssessment, SceneInspector
from liminal.policy.mock import Fault, MockBackend

TASK = "configs/tasks/replace_component.yaml"


def _response(**values: bool | str) -> dict:
    return {
        "predicates": {
            predicate: {"value": value, "evidence": f"fixture says {predicate} is {value}"}
            for predicate, value in values.items()
        }
    }


def _run(make_ctx, responses, backend=None):
    backend = backend or MockBackend(duration_s=0.001)
    ctx = make_ctx(backend)
    ctx.inspector = SceneInspector(
        FixtureVisionModel(responses),
        ctx.cameras,
        ctx.world,
        confirmations=1,
    )
    report = asyncio.run(run_task(load_task(TASK, arms=ctx.arm_names), ctx))
    return report, ctx, backend


def _lid(open_: bool = True):
    return _response(lid_open=open_)


def _scene(*, clear: bool, removed: bool = False, installed: bool = False):
    return _response(
        access_clear=clear,
        access_blocked=not clear,
        faulty_component_in_red_cup=removed,
        replacement_in_slot=installed,
    )


def _home():
    return _response(main_arm_home=True)


def _restored():
    return _response(cable_secured_clear=False, cable_restored=True)


def test_unobstructed_route_uses_only_main_arm_work(make_ctx):
    report, _, backend = _run(make_ctx, [_lid(), _scene(clear=True), _home()])

    assert report.goal_verified
    assert backend.calls == [
        "right:remove_to_red_cup(faulty_component)",
        "right:green_cup_to_slot(green_replacement)",
    ]


def test_obstructed_route_opens_lid_holds_cable_and_restores_it(make_ctx):
    report, ctx, backend = _run(make_ctx, [_lid(False), _scene(clear=False), _home(), _restored()])

    assert report.goal_verified
    assert backend.calls == [
        "left:open_lid(lid)",
        "left:clear_cable_to_hold(cable)",
        "right:remove_to_red_cup(faulty_component)",
        "right:green_cup_to_slot(green_replacement)",
        "left:restore_cable(cable)",
    ]
    assert ctx.world.holds("cable_restored")
    assert ctx.world.holds("!cable_secured_clear")


def test_dropped_cable_cancels_main_work_and_retries_obstructed_route(make_ctx):
    backend = MockBackend(
        duration_s=0.02,
        faults=[Fault("remove_to_red_cup", on_attempt=1, facts={"cable_secured_clear": False})],
    )
    report, ctx, backend = _run(
        make_ctx,
        [_lid(), _scene(clear=False), _home(), _restored()],
        backend,
    )

    assert report.goal_verified
    assert backend.calls.count("left:clear_cable_to_hold(cable)") == 2
    assert backend.calls.count("right:remove_to_red_cup(faulty_component)") == 2
    assert any(event["kind"] == "while_violated" for event in ctx.world.events)


def test_failed_placement_is_retried(make_ctx):
    backend = MockBackend(
        duration_s=0.001,
        faults=[Fault("green_cup_to_slot", on_attempt=1, fail=True)],
    )
    report, _, backend = _run(make_ctx, [_lid(), _scene(clear=True), _home()], backend)

    assert report.goal_verified
    assert backend.calls.count("right:green_cup_to_slot(green_replacement)") == 2


def test_restart_after_removal_skips_the_completed_skill(make_ctx):
    report, _, backend = _run(
        make_ctx,
        [_lid(), _scene(clear=True, removed=True), _home()],
    )

    assert report.goal_verified
    assert "right:remove_to_red_cup(faulty_component)" not in backend.calls
    assert backend.calls == ["right:green_cup_to_slot(green_replacement)"]


def test_ambiguous_access_fails_before_any_manipulation(make_ctx):
    class InconclusiveInspector:
        def __init__(self, world):
            self.world = world
            self.calls = 0

        async def inspect_until_known(self, request, **kwargs):
            self.calls += 1
            if self.calls == 1:
                assessment = SceneAssessment.from_model_output(request, _lid())
            else:
                assessment = SceneAssessment.unknown(request, "access is ambiguous in the fixture")
            for observation in assessment.observations.values():
                self.world.observe(
                    observation.predicate,
                    observation.value,
                    source="inconclusive_fixture",
                    captured_at=assessment.captured_at,
                    evidence=observation.evidence,
                )
            return assessment

    backend = MockBackend(duration_s=0.001)
    ctx = make_ctx(backend)
    ctx.inspector = InconclusiveInspector(ctx.world)
    report = asyncio.run(run_task(load_task(TASK, arms=ctx.arm_names), ctx))

    assert report.status is Status.TIMEOUT
    assert not report.goal_verified
    assert backend.calls == []
