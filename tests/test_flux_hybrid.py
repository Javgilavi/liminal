"""The Claude gate inside the harness's FLUX backend: a fake reasoner and mock arms (no network, GPU or API key)."""

import asyncio
import base64
import sys

import pytest

from liminal.app import build_context
from liminal.core.types import Skill, Status
from liminal.orchestration.bt import SkillNode, While
from liminal.policy.flux import FluxConfig, FluxPolicyBackend
from liminal.policy.hybrid import HarnessHybrid
from liminal.safety.supervisor import JointLimits

pytest.importorskip("numpy")
pytest.importorskip("PIL")

CAMERAS = {"left": {"scene": "top", "wrist": "left_wrist"}, "right": {"scene": "top", "wrist": "right_wrist"}}
CHUNK = [[0.5 * i, 0.0, 0.0, 0.0, 0.0, 0.0] for i in range(4)]
CORRECT = {"enabled": True, "arms": {"right": {"kinematics_verified": True}}}
OBSERVE = {"enabled": True}


def decision(payload, mode, steps):
    pose = payload["current_eef"]
    return {"request_id": payload["request_id"], "mode": mode, "steps": steps, "reason": "test",
            "edit": {"delta_position": [0, 0, .001], "delta_rotation_vector": [0, 0, 0], "gripper": "keep"},
            "target": {"position": pose["position"], "quaternion_wxyz": pose["quaternion_wxyz"],
                       "gripper_closed": False},
            "assessment": {"task_progress": {"verified_completed": [], "currently_attempting": "pick",
                                              "remaining": ["place"]},
                           "current_subgoal": "pick", "execution_status": "failed",
                           "execution_evidence": "empty gripper", "expected_next_intent": "pick",
                           "predicted_next_intent": "place", "intent_status": "misaligned",
                           "intent_evidence": "proposal moves away"}}


class FakeReasoner:
    def __init__(self, mode="student", steps=2, during=None):
        self.mode, self.steps, self.during = mode, steps, during
        self.calls = []
        self.entered = asyncio.Event()

    async def decide(self, payload, images, schema):
        self.calls.append((payload, images))
        self.entered.set()
        if self.during:
            await self.during()
        return decision(payload, self.mode, self.steps)


class GatedFlux(FluxPolicyBackend):
    def __init__(self, reasoner, hybrid, **kw):
        super().__init__(FluxConfig(**{"chunk_size": 4, "execute_steps": 3, "control_hz": 1000.0,
                                       "camera_map": CAMERAS, "hybrid": hybrid, **kw}))
        self.reasoner = reasoner

    def load(self):
        self._model = object()

    def predict(self, history, instruction, target):
        return [list(action) for action in CHUNK]

    def hybrid_reasoner(self, target, config):
        return self.reasoner


def context(backend, limits=True):
    ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=backend)
    if limits:
        for arm in ("left", "right"):
            ctx.safety.limits[arm] = JointLimits([-100.0] * 5 + [0.0], [100.0] * 6, max_step=4.0)
    return ctx


def kinds(ctx):
    return [event["kind"] for event in ctx.world.events]


def run(backend, ctx, arm="right", **kw):
    return asyncio.run(backend.execute(Skill(arm, "pick_place", prompt="put the pen in the slot", **kw), ctx))


@pytest.mark.parametrize("hybrid", [
    {},
    {"config": "configs/hybrid/claude_opus_haller.yaml"},  # that file's enabled: true does not switch it on
    {"enabled": False, "arms": {"right": {"kinematics_verified": True}}},
])
def test_gate_is_off_unless_the_policy_config_enables_it(monkeypatch, hybrid):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    backend = GatedFlux(FakeReasoner("stop"), hybrid)
    ctx = context(backend)
    assert run(backend, ctx).status is Status.SUCCESS
    assert backend._hybrid == {} and backend.reasoner.calls == []
    assert ctx.arms["right"].history == CHUNK[:3]
    shipped = HarnessHybrid.parse({"config": "configs/hybrid/claude_opus_haller.yaml"}).config("right")
    assert shipped.model == "claude-opus-5-5" and not shipped.enabled
    # Lazy: a per-arm build with the gate on imports no model SDK and builds no controller.
    ctx = build_context("configs/robot/mock_dual.yaml", backend="flux",
                        policy={"instances": "per_arm", "hybrid": {"enabled": True}})
    assert all(policy._hybrid == {} for policy in ctx.skills.arm_policies.values())


def test_continue_runs_the_student_prefix_through_the_supervisor():
    backend = GatedFlux(FakeReasoner("student", steps=2), CORRECT)
    ctx = context(backend)
    assert run(backend, ctx).status is Status.SUCCESS
    assert ctx.arms["right"].history == CHUNK[:2]  # the chosen prefix of 3 executable actions, the rest discarded
    assert ctx.arms["left"].history == []
    (payload, images), = backend.reasoner.calls
    assert payload["task"] == "put the pen in the slot" and payload["student_joints"] == CHUNK
    assert payload["current_joints"] == [0.0] * 6 and payload["max_student_steps"] == 3
    assert [image["label"] for image in images] == ["current_scene", "current_wrist"]
    assert all(base64.b64decode(image["data"])[:2] == b"\xff\xd8" for image in images)  # JPEG
    session = next(e for e in ctx.world.events if e["kind"] == "hybrid_session")
    assert session["mode"] == "correct" and session["arm"] == "right"
    request = next(e for e in ctx.world.events if e["kind"] == "hybrid_request")
    assert request["images"] == ["current_scene", "current_wrist"]  # labels only, never the frames
    assert {"hybrid_decision", "hybrid_executed"} <= set(kinds(ctx))
    assert backend._hybrid["right"].history[-1]["executed_joints"] == CHUNK[:2]


@pytest.mark.parametrize("mode", ["eef", "edit"])
def test_correction_is_bounded_local_ik_sent_through_the_supervisor(mode):
    backend = GatedFlux(FakeReasoner(mode, steps=2), CORRECT)
    ctx = context(backend)
    sent = []
    original = ctx.safety.send

    def spy(arm, target):
        safe = original(arm, target)
        sent.append((arm, list(target), safe))
        return safe

    ctx.safety.send = spy
    assert run(backend, ctx).status is Status.SUCCESS
    history = ctx.arms["right"].history
    assert len(history) == 2 and history == [safe for _, _, safe in sent]
    assert all(arm == "right" for arm, _, _ in sent)
    if mode == "eef":
        # Hold the current pose and open the gripper: IK asks +10 % per control, the supervisor allows 4.
        assert [target[5] for _, target, _ in sent] == pytest.approx([10, 14])
        assert [q[5] for q in history] == pytest.approx([4, 8])
        assert all(q[:5] == pytest.approx([0.0] * 5) for q in history)
    assert backend._hybrid["right"].history[-1]["decision"]["mode"] == mode


def test_stop_ends_the_attempt_without_motion_or_claimed_success():
    backend = GatedFlux(FakeReasoner("stop"), CORRECT)
    ctx = context(backend)
    result = run(backend, ctx, success="object_placed")
    assert result.status is Status.FAILURE and "stopped" in result.message
    assert ctx.arms["right"].history == [] and ctx.arms["right"].stopped
    assert ctx.world.evaluate("object_placed") is not True


@pytest.mark.parametrize("mode", ["eef", "stop"])
def test_unverified_kinematics_only_observes(mode):
    backend = GatedFlux(FakeReasoner(mode, steps=2), OBSERVE)
    ctx = context(backend, limits=False)
    assert run(backend, ctx).status is Status.SUCCESS
    assert ctx.arms["right"].history == CHUNK[:3]  # FLUX unchanged
    session = next(e for e in ctx.world.events if e["kind"] == "hybrid_session")
    assert session["mode"] == "observe" and session["model"] == "claude-sonnet-5-5"
    decided = next(e for e in ctx.world.events if e["kind"] == "hybrid_decision")
    assert decided["decision"]["mode"] == mode
    assert backend._hybrid["right"].history[-1]["executed_joints"] == CHUNK[:3]


def test_cancelled_skill_while_claude_reasons_sends_nothing():
    async def scenario():
        reasoner = FakeReasoner("eef", during=lambda: asyncio.Event().wait())
        backend = GatedFlux(reasoner, CORRECT)
        ctx = context(backend)
        task = asyncio.create_task(backend.execute(Skill("right", "pick_place", prompt="p"), ctx))
        await asyncio.wait_for(reasoner.entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return ctx

    ctx = asyncio.run(scenario())
    assert ctx.arms["right"].history == [] and ctx.arms["right"].stopped
    error = next(e for e in ctx.world.events if e["kind"] == "hybrid_error")
    assert error["error"] == "CancelledError" and "flux_cancelled" in kinds(ctx)


def test_while_invariant_breaking_mid_decision_cancels_it():
    async def scenario():
        reasoner = FakeReasoner("eef", during=lambda: asyncio.Event().wait())
        backend = GatedFlux(reasoner, CORRECT)
        ctx = context(backend)
        ctx.world.set("lid_open", True, source="test")
        tree = While("left", "lid_open", SkillNode(Skill("right", "pick_place", prompt="p")))
        running = asyncio.create_task(tree.run(ctx))
        await asyncio.wait_for(reasoner.entered.wait(), 5)
        ctx.world.set("lid_open", False, source="test")
        return await asyncio.wait_for(running, 5), ctx

    status, ctx = asyncio.run(scenario())
    assert status is Status.FAILURE and "while_violated" in kinds(ctx)
    assert ctx.arms["right"].history == [] and ctx.arms["right"].stopped


def test_invariant_broken_while_claude_reasons_drops_the_correction():
    async def scenario():
        backend = GatedFlux(FakeReasoner("eef"), CORRECT)
        ctx = context(backend)

        async def lid_falls():
            ctx.world.set("lid_open", False, source="test")

        backend.reasoner.during = lid_falls
        ctx.world.set("lid_open", True, source="test")
        with ctx.watch_invariant("lid_open"):
            result = await backend.execute(Skill("right", "pick_place", prompt="p"), ctx)
        return result, ctx

    result, ctx = asyncio.run(scenario())
    assert result.status is Status.FAILURE and "lid_open" in result.message
    assert ctx.arms["right"].history == []
    dropped = next(e for e in ctx.world.events if e["kind"] == "hybrid_dropped")
    assert dropped["reason"] == "invariant" and dropped["decision"]["mode"] == "eef"


def test_arm_moved_while_claude_reasons_refuses_the_stale_decision():
    backend = GatedFlux(FakeReasoner("eef"), CORRECT)
    ctx = context(backend)

    async def pushed():
        ctx.arms["right"].state = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    backend.reasoner.during = pushed
    result = run(backend, ctx)
    assert result.status is Status.FAILURE and "stale" in result.message
    assert ctx.arms["right"].history == []


def test_each_arm_has_its_own_gate_history_and_joint_mapping():
    mapped = {"joint_signs": [1, -1, 1, 1, 1], "joint_offsets_deg": [0, 90, 90, 0, 0]}
    backend = GatedFlux(FakeReasoner("student"), {"enabled": True, "arms": {"left": mapped}})
    ctx = context(backend, limits=False)
    assert run(backend, ctx, arm="left").status is Status.SUCCESS
    assert run(backend, ctx, arm="right").status is Status.SUCCESS
    left, right = backend._hybrid["left"], backend._hybrid["right"]
    assert left is not right and left._lock is not right._lock
    assert left.kinematics.offsets.tolist() == [0, 90, 90, 0, 0] and right.kinematics.offsets.tolist() == [0] * 5
    (left_payload, _), (right_payload, _) = backend.reasoner.calls
    assert left_payload["current_eef"]["position"] != pytest.approx(right_payload["current_eef"]["position"])
    assert len(left.history) == len(right.history) == 1
    assert ctx.arms["left"].history == ctx.arms["right"].history == CHUNK[:3]
    backend.reset("left")
    assert left.history == [] and left.previous_images == [] and len(right.history) == 1


@pytest.mark.parametrize("arm, hybrid, camera_map, limits, message", [
    ("both", OBSERVE, None, True, "6-joint"),
    ("right", OBSERVE, {"right": {"scene": "top"}}, True, "scene and wrist"),
    ("right", CORRECT, None, False, "joint limits"),
])
def test_unusable_gate_targets_fail_before_motion(arm, hybrid, camera_map, limits, message):
    kw = {"action_dim": 12} if arm == "both" else {}
    backend = GatedFlux(FakeReasoner(), hybrid, **kw, **({"camera_map": camera_map} if camera_map else {}))
    ctx = context(backend, limits=limits)
    result = run(backend, ctx, arm=arm)
    assert result.status is Status.FAILURE and message in result.message
    assert backend.reasoner.calls == [] and backend._model is None
    assert ctx.arms["left"].history == ctx.arms["right"].history == []


@pytest.mark.parametrize("hybrid", [
    {"enabled": True, "arms": {"right": {"joint_signs": [2, 1, 1, 1, 1]}}},
    {"enabled": True, "temperature": 0.2},
    {"enabled": True, "arms": ["right"]},
    "configs/hybrid/claude_opus_haller.yaml",
])
def test_bad_hybrid_mapping_fails_when_the_backend_is_built(hybrid):
    with pytest.raises(ValueError):
        FluxPolicyBackend(FluxConfig(hybrid=hybrid))
