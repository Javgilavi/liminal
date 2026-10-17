import asyncio

import pytest

from liminal.bimanual import ArmBase, NoFeasiblePlan, Scene, SceneObject, plan_clear_and_act
from liminal.data.protocol import make_plan
from liminal.hardware import MockArm
from liminal.orchestration import parse_task, run_task
from liminal.perception.world_state import WorldState
from liminal.safety import EmergencyStop, JointLimits, SafetySupervisor
from liminal.sim.embodiment import Embodiment


def test_world_state_negation_and_log(tmp_path):
    w = WorldState(log_path=tmp_path / "e.jsonl")
    assert not w.holds("lid_open") and not w.holds("!lid_open")
    w.set("lid_open", True, source="test")
    assert w.holds("lid_open") and not w.holds("!lid_open")
    assert (tmp_path / "e.jsonl").read_text().count("lid_open") == 1


def test_safety_clips_limits_and_step():
    arm = MockArm("a")
    s = SafetySupervisor({"a": arm}, {"a": JointLimits([-10] * 6, [10] * 6, max_step=2)})
    out = s.send("a", [50, -50, 1, 0, 0, 0])
    assert out == [2, -2, 1, 0, 0, 0]
    with pytest.raises(ValueError):
        s.send("a", [float("nan")] * 6)


def test_estop_blocks_actions():
    arm = MockArm("a")
    s = SafetySupervisor({"a": arm})
    s.estop()
    assert arm.stopped
    with pytest.raises(EmergencyStop):
        s.send("a", [0] * 6)


def _scene(strategy):
    return Scene(
        arms=[ArmBase("left", (-0.2, 0.0)), ArmBase("right", (0.2, 0.0))],
        objects=[SceneObject("cup", (-0.05, 0.15), strategy, blocks=["block"]), SceneObject("block", (0.05, 0.2))],
    )


@pytest.mark.parametrize("strategy", ["push_aside", "pick_aside", "hold_away"])
def test_bimanual_planner_output_runs(strategy, make_ctx):
    spec_dict = plan_clear_and_act(_scene(strategy), "block", "Put the block in the bowl", goal="block_in_bowl")
    spec = parse_task(spec_dict, arms=["left", "right"])
    report = asyncio.run(run_task(spec, make_ctx()))
    assert report.goal_verified
    steps = spec_dict["tree"]["sequence"]
    if strategy == "hold_away":
        assert any("while" in s for s in steps)


def test_bimanual_planner_uses_two_arms():
    steps = plan_clear_and_act(_scene("push_aside"), "block", "p", goal="g")["tree"]["sequence"]
    clear_arm = steps[0]["retry"]["do"]["skill"]["arm"]
    act_arm = steps[1]["retry"]["do"]["skill"]["arm"]
    assert {clear_arm, act_arm} == {"left", "right"}


def test_bimanual_hold_needs_two_arms():
    scene = _scene("hold_away")
    scene.arms = [ArmBase("right", (0.2, 0.0), reach_m=1.0)]
    with pytest.raises(NoFeasiblePlan):
        plan_clear_and_act(scene, "block", "p", goal="g")


def test_xlerobot_embodiment_split_merge():
    e = Embodiment.load("configs/embodiment/xlerobot.yaml")
    assert e.action_dim == 12
    parts = e.split(list(range(12)))
    assert parts["right_arm"] == list(range(6, 12))
    assert e.merge(parts) == list(range(12))


def test_demo_plan_covers_variations():
    plans = make_plan(60, ["Pick up the red block and place it in the bowl"], zones=["A", "B", "C"])
    assert len(plans) == 60
    assert {p.object_zone for p in plans} == {"A", "B", "C"}


def test_claude_md_is_a_copy_of_agent_md():
    from pathlib import Path

    # The checkout migrated the instruction filename to AGENTS.md before WS3 work.
    instructions = Path("AGENTS.md") if Path("AGENTS.md").exists() else Path("AGENT.md")
    assert Path("CLAUDE.md").read_text() == instructions.read_text(), "keep agent instruction copies in sync"
