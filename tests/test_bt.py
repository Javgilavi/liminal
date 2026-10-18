import asyncio

import pytest

from liminal.core.types import Skill, SkillResult, Status
from liminal.orchestration import bt, parse_task
from liminal.perception import FixtureVisionModel, SceneInspector
from liminal.policy.mock import Fault, MockBackend
from liminal.safety import ArmConflict


def run(coro):
    return asyncio.run(coro)


def test_sequence_stops_at_first_failure(make_ctx):
    backend = MockBackend(duration_s=0.01, faults=[Fault("b", fail=True)])
    ctx = make_ctx(backend)
    tree = bt.Sequence([bt.SkillNode(Skill("left", n)) for n in ("a", "b", "c")])
    assert run(tree.run(ctx)) is Status.FAILURE
    assert backend.calls == ["left:a", "left:b"]


def test_skill_success_is_verified_independently(make_ctx):
    class LiarBackend:
        async def execute(self, skill, ctx):
            from liminal.core.types import SkillResult

            return SkillResult(Status.SUCCESS, skill)  # claims success, changes nothing

    ctx = make_ctx(LiarBackend())
    node = bt.SkillNode(Skill("right", "pick_place", success="block_in_bowl", verify_timeout_s=0.05))
    assert run(node.run(ctx)) is Status.FAILURE


def test_skill_timeout_stops_arm(make_ctx):
    class SlowBackend:
        async def execute(self, skill, ctx):
            await asyncio.sleep(10)

    ctx = make_ctx(SlowBackend())
    node = bt.SkillNode(Skill("right", "pick_place", timeout_s=0.05))
    assert run(node.run(ctx)) is Status.TIMEOUT
    assert ctx.arms["right"].stopped


def test_parallel_same_arm_is_a_conflict(make_ctx):
    ctx = make_ctx()
    tree = bt.Parallel([bt.SkillNode(Skill("left", "a")), bt.SkillNode(Skill("left", "b"))])
    with pytest.raises(ArmConflict):
        run(tree.run(ctx))


def test_parallel_two_arms(make_ctx, mock_backend):
    ctx = make_ctx()
    tree = bt.Parallel([bt.SkillNode(Skill("left", "a")), bt.SkillNode(Skill("right", "b"))])
    assert run(tree.run(ctx)) is Status.SUCCESS
    assert sorted(mock_backend.calls) == ["left:a", "right:b"]


def test_while_holds_and_requires_until(make_ctx):
    ctx = make_ctx()
    ctx.world.set("lid_open", True)
    body = bt.SkillNode(Skill("right", "pick_place", success="object_inserted"))
    node = bt.While("left", maintain="lid_open", until="object_inserted", body=body)
    assert run(node.run(ctx)) is Status.SUCCESS
    assert ctx.arms["left"].holding is not None


def test_while_registers_active_invariant(make_ctx):
    class Backend:
        active = ()

        async def execute(self, skill, ctx):
            self.active = ctx.active_invariants
            return SkillResult(Status.SUCCESS, skill)

    backend = Backend()
    ctx = make_ctx(backend)
    ctx.world.set("cable_secured_clear", True)
    node = bt.While(
        "left",
        maintain="cable_secured_clear",
        body=bt.SkillNode(Skill("right", "remove_to_red_cup")),
    )
    assert run(node.run(ctx)) is Status.SUCCESS
    assert backend.active == ("cable_secured_clear",)
    assert ctx.active_invariants == ()


def test_while_cancels_body_when_invariant_breaks(make_ctx):
    backend = MockBackend(duration_s=0.2, faults=[Fault("pick_place", facts={"lid_open": False})])
    ctx = make_ctx(backend)
    ctx.world.set("lid_open", True)
    body = bt.SkillNode(Skill("right", "pick_place", success="object_inserted"))
    node = bt.While("left", maintain="lid_open", body=body)
    assert run(node.run(ctx)) is Status.FAILURE
    assert not ctx.world.holds("object_inserted")
    assert any(e["kind"] == "while_violated" for e in ctx.world.events)


def test_while_body_cannot_use_support_arm(make_ctx):
    ctx = make_ctx()
    ctx.world.set("lid_open", True)
    node = bt.While("left", maintain="lid_open", body=bt.SkillNode(Skill("left", "pick_place")))
    with pytest.raises(ArmConflict):
        run(node.run(ctx))


def test_while_fails_if_condition_never_established(make_ctx):
    ctx = make_ctx()
    node = bt.While("left", maintain="lid_open", body=bt.Check("x"), establish_timeout_s=0.05)
    assert run(node.run(ctx)) is Status.FAILURE


def test_retry_runs_recovery_between_attempts(make_ctx):
    backend = MockBackend(duration_s=0.01, faults=[Fault("pick_place", fail=True)])
    ctx = make_ctx(backend)
    node = bt.Retry(bt.SkillNode(Skill("right", "pick_place")), attempts=2,
                    recovery=bt.SkillNode(Skill("right", "reset_pose")))
    assert run(node.run(ctx)) is Status.SUCCESS
    assert backend.calls == ["right:pick_place", "right:reset_pose", "right:pick_place"]


def test_waitfor_and_timeout(make_ctx):
    ctx = make_ctx()

    async def scenario():
        async def later():
            await asyncio.sleep(0.02)
            ctx.world.set("ready", True)

        asyncio.get_running_loop().create_task(later())
        ok = await bt.WaitFor("ready", timeout_s=1).run(ctx)
        late = await bt.Timeout(bt.WaitFor("never", timeout_s=None), seconds=0.05).run(ctx)
        return ok, late

    assert run(scenario()) == (Status.SUCCESS, Status.TIMEOUT)


def test_inspect_node_waits_for_known_consensus(make_ctx):
    response = {"predicates": {"lid_open": {"value": "false", "evidence": "lid is down"}}}
    model = FixtureVisionModel([response, response])
    ctx = make_ctx()
    ctx.inspector = SceneInspector(model, ctx.cameras, ctx.world, confirmations=2)
    spec = parse_task({
        "tree": {
            "inspect": {
                "predicates": ["lid_open"],
                "cameras": ["top"],
                "definitions": {"lid_open": "The lid is raised and the box interior is visible."},
                "context": "Inspect the mock server box.",
                "timeout_s": 0.2,
                "retry_delay_s": 0,
            }
        }
    })
    assert run(spec.tree.run(ctx)) is Status.SUCCESS
    assert ctx.world.known("lid_open") and not ctx.world.holds("lid_open")
    assert "box interior is visible" in model.calls[0][0]


def test_group_skill_claims_every_member_arm(make_ctx, mock_backend):
    ctx = make_ctx()
    assert run(bt.SkillNode(Skill("both", "open_lid_and_insert")).run(ctx)) is Status.SUCCESS
    assert mock_backend.calls == ["both:open_lid_and_insert"]
    clash = bt.Parallel([bt.SkillNode(Skill("both", "x")), bt.SkillNode(Skill("left", "y"))])
    with pytest.raises(ArmConflict):
        run(clash.run(ctx))


def test_group_primitive_moves_both_arms(make_ctx):
    ctx = make_ctx()
    for a in ctx.arms.values():
        a.state = [5.0] * 6
    assert run(bt.SkillNode(Skill("both", "home")).run(ctx)) is Status.SUCCESS
    assert all(a.state == [0.0] * 6 for a in ctx.arms.values())


def test_failed_recovery_prevents_another_attempt(make_ctx):
    backend = MockBackend(duration_s=0, faults=[Fault("work", fail=True), Fault("recover", fail=True)])
    ctx = make_ctx(backend)
    node = bt.Retry(bt.SkillNode(Skill("right", "work")), attempts=2,
                    recovery=bt.SkillNode(Skill("right", "recover")))
    assert run(node.run(ctx)) is Status.FAILURE
    assert backend.calls == ["right:work", "right:recover"]


def test_cancel_retains_ownership_until_arm_is_stopped(make_ctx):
    class Waiting:
        async def execute(self, skill, ctx):
            await asyncio.Event().wait()

    ctx = make_ctx(Waiting())
    arm = ctx.arms["right"]
    original_stop = arm.stop
    owners = []

    def stop():
        owners.append(ctx.safety.owner_of("right"))
        original_stop()

    arm.stop = stop

    async def scenario():
        task = asyncio.create_task(bt.SkillNode(Skill("right", "work")).run(ctx))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(scenario())
    assert owners == ["right:work"]
    assert ctx.safety.owner_of("right") is None


def test_stale_true_does_not_verify_skill_with_inspector(make_ctx):
    class Liar:
        async def execute(self, skill, ctx):
            return SkillResult(Status.SUCCESS, skill)

    ctx = make_ctx(Liar())
    ctx.inspector = object()
    ctx.world.observe("done", True, source="old")
    assert run(bt.SkillNode(Skill("right", "work", success="done", verify_timeout_s=0.001)).run(ctx)) is Status.FAILURE


def test_invariant_expires_without_another_world_update(make_ctx):
    async def scenario():
        ctx = make_ctx()
        ctx.world.observe("held", True, source="fixture")
        await asyncio.wait_for(ctx.world.wait_until_violated("held", max_age_s=0.01), 0.1)
        assert ctx.world.holds("held")  # expiry, not a new false update, woke the monitor

    run(scenario())


def test_late_result_from_old_capture_cannot_verify_new_skill(make_ctx):
    import time

    captured = time.monotonic() - 10

    class OldResult:
        async def execute(self, skill, ctx):
            ctx.world.observe("done", True, source="delayed", captured_at=captured)
            return SkillResult(Status.SUCCESS, skill)

    ctx = make_ctx(OldResult())
    ctx.inspector = object()
    node = bt.SkillNode(Skill("right", "work", success="done", verify_timeout_s=0.001))
    assert run(node.run(ctx)) is Status.FAILURE


def test_old_capture_does_not_overwrite_new_false_evidence(make_ctx):
    import time

    ctx = make_ctx()
    ctx.world.observe("done", False, source="new")
    ctx.world.observe("done", True, source="delayed", captured_at=time.monotonic() - 1)
    assert not ctx.world.holds("done")


def test_while_refreshes_invariant_during_slow_body(make_ctx):
    from liminal.perception.types import InspectionProfile

    class Slow:
        async def execute(self, skill, ctx):
            await asyncio.sleep(0.08)
            return SkillResult(Status.SUCCESS, skill)

    ctx = make_ctx(Slow())
    ctx.inspection = InspectionProfile(definitions={"held": "support holds object"}, timeout_s=0.03)

    class Inspector:
        calls = 0

        async def inspect_until_known(self, request, **kwargs):
            self.calls += 1
            ctx.world.observe("held", True, source="refresh")

    ctx.inspector = Inspector()
    ctx.world.observe("held", True, source="initial")
    node = bt.While("left", "held", bt.SkillNode(Skill("right", "slow")))
    assert run(node.run(ctx)) is Status.SUCCESS
    assert ctx.inspector.calls >= 2


def test_invariant_violation_remains_failure_if_refreshed_during_cleanup(make_ctx):
    ctx = make_ctx()
    ctx.world.set("held", True)

    async def expired_then_refreshed(condition, **kwargs):
        ctx.world.set("held", False)
        ctx.world.set("held", True)

    ctx.world.wait_until_violated = expired_then_refreshed
    node = bt.While("left", "held", bt.SkillNode(Skill("right", "work")))
    assert run(node.run(ctx)) is Status.FAILURE
