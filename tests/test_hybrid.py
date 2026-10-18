"""Action gate tests: no robot, GPU, network, or model API."""
import asyncio
import copy
import math
import sys
from types import SimpleNamespace

import pytest

from liminal.hardware.kinematics import SO101Kinematics
from liminal.policy.hybrid import (
    ClaudeReasoner,
    HybridConfig,
    HybridController,
    build_hybrid,
    decision_schema,
    validate_schema,
)


def kinematics():
    pytest.importorskip("numpy")
    return SO101Kinematics("sim/single_arm/so101.xml", [1]*5, [0]*5)


def decision(payload, mode="student", steps=2):
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


class Reasoner:
    def __init__(self, mode="student", change=None):
        self.mode, self.change = mode, change
        self.calls = []

    async def decide(self, payload, images, schema):
        self.calls.append((copy.deepcopy(payload), copy.deepcopy(images)))
        result = decision(payload, self.mode)
        if self.change:
            self.change(result)
        return result


def controller(mode="student", change=None, **kw):
    return HybridController(HybridConfig(enabled=True, **kw), reasoner=Reasoner(mode, change), kinematics=kinematics())


def choose(c, actions=None):
    return c.choose(goal="put pen in slot", actions=actions or [[0, 0, 0, 0, 0, 50]]*42,
                    observation={"state": [0, 0, 0, 0, 0, 50], "scene": "eA==", "wrist": "eA=="},
                    max_steps=32, log=lambda *a, **kw: None)


def test_disabled_has_no_optional_imports_or_model_initialization(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert build_hybrid({}, execute=True) is None
    assert build_hybrid({"hybrid": {"enabled": False}}, execute=True) is None


def test_unverified_mapping_blocks_before_loading_model():
    with pytest.raises(ValueError, match="verified"):
        build_hybrid({"hybrid": {"enabled": True, "model_path": "/missing"}}, execute=True)


def test_fk_zero_pose_regression_and_calibrated_mapping():
    kin = kinematics()
    pose = kin.forward([0, 0, 0, 0, 0, 50])
    assert pose["position"] == pytest.approx([.3913615, -.0009794652485835525, .2463451766930268])
    mapped = SO101Kinematics("sim/single_arm/so101.xml", [1, -1, 1, 1, 1], [0, 90, 90, 0, 0])
    q = [12, -25, 40, 10, -20, 60]
    mapped_q = [12, 115, 130, 10, -20, 60]
    assert mapped.forward(mapped_q)["position"] == pytest.approx(kin.forward(q)["position"])
    assert mapped.forward(mapped_q)["quaternion_wxyz"] == pytest.approx(kin.forward(q)["quaternion_wxyz"])


def test_rotation_round_trip_including_pi():
    kin = kinematics()
    for v in ([0, 0, 0], [.2, -.3, .1], [math.pi, 0, 0], [0, math.pi, 0]):
        r = kin.exp(v)
        assert kin.rotation(kin.quaternion(r)) == pytest.approx(r)
        assert kin.exp(kin.rotvec(r)) == pytest.approx(r)


def test_local_ik_is_bounded_and_does_not_claim_arrival():
    kin = kinematics()
    q = [0, -30, 20, 15, 0, 50]
    target = kin.forward(q)
    target["position"][2] += .01
    target["gripper_percent"] = 100
    next_q = kin.step(q, target, [-100]*5+[0], [100]*6, max_joint_step_deg=math.degrees(.05), max_gripper_step=2)
    assert all(abs(a-b) <= math.degrees(.05)+1e-8 for a, b in zip(q[:5], next_q[:5], strict=True))
    assert next_q[-1] == 52
    assert math.dist(kin.forward(next_q)["position"], target["position"]) < .01


@pytest.mark.parametrize("mode", ["student", "edit", "eef", "stop"])
def test_all_four_modes(mode):
    c = controller(mode)
    segment = asyncio.run(choose(c))
    assert segment.stopped == (mode == "stop")
    assert segment.steps == (0 if mode == "stop" else 2)
    payload, images = c.reasoner.calls[0]
    assert len(payload["student_eef"]) == 42 and payload["task"] == "put pen in slot"
    assert {i["label"] for i in images} == {"current_scene", "current_wrist"}
    if mode != "stop":
        target = c.target(segment, 0, segment.measured, [-100]*5+[0], [100]*6)
        assert len(target) == 6
    if mode == "student":
        assert target == [0, 0, 0, 0, 0, 50]
    elif mode == "edit":
        original = c.kinematics.forward(segment.joints[0])
        assert segment.poses[0]["position"][2] - original["position"][2] == pytest.approx(.0005)
        assert segment.poses[1]["position"][2] - original["position"][2] == pytest.approx(.001)
    elif mode == "eef":
        assert segment.poses[0] == segment.poses[1]


@pytest.mark.parametrize("change", [
    lambda d: d.update(request_id="stale"),
    lambda d: d.update(steps=16),
    lambda d: d.update(steps=True),
    lambda d: d.update(mode="home"),
    lambda d: d.update(success=True),
    lambda d: d["edit"].update(delta_position=[0, 0, float("nan")]),
    lambda d: d["edit"].update(delta_position=[0, 0, .1]),
    lambda d: d["assessment"].update(execution_status="uncertain", intent_status="uncertain"),
])
def test_invalid_or_unjustified_decision_is_rejected(change):
    c = controller("edit", change)
    with pytest.raises(ValueError):
        asyncio.run(choose(c))


def test_five_step_takeover_limit_and_distant_target():
    for mode, change in [("edit", lambda d: d.update(steps=6)),
                         ("eef", lambda d: d["target"].update(position=[10, 0, 0])),
                         ("eef", lambda d: d["target"].update(quaternion_wxyz=[2, 0, 0, 0]))]:
        with pytest.raises(ValueError):
            asyncio.run(choose(controller(mode, change)))


def test_selected_prefix_only_travel_validation():
    c = controller()
    actions = [[0, 0, 0, 0, 0, 50]]*42
    actions[-1] = [999]*6
    segment = asyncio.run(choose(c, actions))
    assert c.target(segment, 0, segment.measured, [-100]*5+[0], [100]*6) == actions[0]
    actions[0] = [999]*6
    segment = asyncio.run(choose(c, actions))
    with pytest.raises(ValueError, match="travel"):
        c.target(segment, 0, segment.measured, [-100]*5+[0], [100]*6)


def test_stale_state_and_each_step_uses_measured_feedback():
    c = controller("eef")
    segment = asyncio.run(choose(c))
    moved = [5, 0, 0, 0, 0, 50]
    with pytest.raises(ValueError, match="stale"):
        c.validate_fresh(segment, moved)
    one = c.target(segment, 0, segment.measured, [-100]*5+[0], [100]*6)
    two = c.target(segment, 1, [1, 0, 0, 0, 0, 50], [-100]*5+[0], [100]*6)
    assert one != two


def test_history_records_actual_commands_and_prior_rgb():
    c = controller(history_turns=1)
    segment = asyncio.run(choose(c))
    c.record(segment, [[1]*6], [2]*6, lambda *a, **kw: None)
    asyncio.run(choose(c))
    payload, images = c.reasoner.calls[-1]
    assert payload["episode_history"][0]["executed_joints"] == [[1]*6]
    assert payload["episode_history"][0]["after"] == [2]*6
    assert len(images) == 4
    assert images[0]["label"] == "previous_before_scene"


def test_reasoner_timeout_and_cancellation_propagate():
    async def run(cancel):
        c = controller(timeout_s=.01)
        entered = asyncio.Event()
        class Waiting:
            async def decide(self, *args):
                entered.set()
                await asyncio.Event().wait()
        c.reasoner = Waiting()
        task = asyncio.create_task(choose(c))
        await entered.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await task
    asyncio.run(run(False))
    asyncio.run(run(True))


def test_claude_multimodal_tool_contract_no_network(monkeypatch):
    seen = {}
    class Client:
        def __init__(self, **kw):
            seen.update(kw)
            self.messages = self
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def create(self, **kw):
            seen.update(kw)
            return SimpleNamespace(
                stop_reason="tool_use", usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 10}),
                content=[SimpleNamespace(type="tool_use", name="hybrid_decision", input={"ok": True})])
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(AsyncAnthropic=Client))
    result = asyncio.run(ClaudeReasoner(HybridConfig()).decide(
        {"task": "pick"}, [{"label": "scene", "data": "eA=="}], decision_schema("id")))
    assert result == {"ok": True}
    assert seen["max_retries"] == 0 and seen["timeout"] == 30
    assert seen["tool_choice"] == {"type": "tool", "name": "hybrid_decision"}
    assert seen["messages"][0]["content"][2]["type"] == "image"


def test_schema_rejects_nonfinite_unused_fields():
    data = decision({"request_id": "a", "current_eef": {"position": [0]*3, "quaternion_wxyz": [1, 0, 0, 0]}})
    data["target"]["position"][0] = float("inf")
    with pytest.raises(ValueError):
        validate_schema(data, decision_schema("a"))


def test_fk_matches_mujoco_robot_only_oracle_when_available():
    """Independent FK oracle; strip meshes/physics assets, never step simulation."""
    import xml.etree.ElementTree as ET

    np = pytest.importorskip("numpy")
    mj = pytest.importorskip("mujoco")
    root = ET.parse("sim/single_arm/so101.xml").getroot()
    for child in list(root):
        if child.tag not in ("compiler", "worldbody"):
            root.remove(child)
    for element in root.iter():
        element.attrib.pop("class", None)
        element.attrib.pop("childclass", None)
        for child in list(element):
            if child.tag not in ("compiler", "worldbody", "body", "joint", "site", "inertial"):
                element.remove(child)
    model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mj.MjData(model)
    site = model.site("gripperframe").id
    kin = kinematics()
    from liminal.core.types import SO101_JOINTS
    rng = np.random.default_rng(718)
    for _ in range(100):
        q = rng.uniform(-60, 60, 5).tolist()+[50.]
        for name, value in zip(SO101_JOINTS[:5], q[:5], strict=True):
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = math.radians(value)
        mj.mj_forward(model, data)
        ours = kin.matrix(q)
        assert ours[:3, 3] == pytest.approx(data.site_xpos[site], abs=1e-10)
        assert ours[:3, :3] == pytest.approx(data.site_xmat[site].reshape(3, 3), abs=1e-10)
