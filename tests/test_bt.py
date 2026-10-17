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
