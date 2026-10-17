"""End-to-end runs of the shipped task files on mock arms."""

import asyncio
from pathlib import Path

import pytest
import yaml

from liminal.core.types import Status
from liminal.orchestration import TaskSpecError, load_task, parse_task, run_task
from liminal.policy.mock import Fault, MockBackend

TASKS = sorted(Path("configs/tasks").glob("*.yaml"))


@pytest.mark.parametrize("path", TASKS, ids=lambda p: p.stem)
def test_shipped_tasks_succeed_on_mock(path, make_ctx):
    ctx = make_ctx()
    report = asyncio.run(run_task(load_task(path, arms=ctx.arm_names), ctx))
    assert report.goal_verified, report


def test_lid_slip_triggers_recovery_and_retry(make_ctx):
    backend = MockBackend(duration_s=0.05, faults=[Fault("pick_place", on_attempt=1, facts={"lid_open": False})])
    ctx = make_ctx(backend)
    report = asyncio.run(run_task(load_task("configs/tasks/lid_hold_insert.yaml"), ctx))
    assert report.goal_verified
    assert backend.calls.count("left:open_lid(lid)") == 2  # reopened after the slip
    kinds = [e["kind"] for e in ctx.world.events]
    assert "while_violated" in kinds and "retry" in kinds


def test_goal_not_reached_is_reported_as_failure(make_ctx):
    ctx = make_ctx()
    spec = parse_task({"name": "t", "goal": "never", "tree": {"skill": {"arm": "left", "name": "home"}}})
    report = asyncio.run(run_task(spec, ctx))
    assert report.status is Status.FAILURE and not report.goal_verified


@pytest.mark.parametrize("bad, msg", [
    ({"tree": {"skil": {}}}, "unknown node type"),
    ({"tree": {"skill": {"arm": "middle", "name": "x"}}}, "unknown arm"),
    ({"tree": {"skill": {"arm": "left"}}}, "'name' is required"),
    ({"tree": {"while": {"arm": "left", "do": {"check": "x"}}}}, "needs arm, maintain"),
    ({"tree": {"sequence": []}}, "non-empty list"),
])
def test_task_spec_errors_are_actionable(bad, msg):
    with pytest.raises(TaskSpecError, match=msg):
        parse_task(bad, arms=["left", "right"])


def test_llm_planner_repairs_invalid_plan():
    from liminal.orchestration.planner import LLMPlanner

    good = yaml.safe_dump({"name": "p", "goal": "block_in_bowl", "tree": {"sequence": [
        {"skill": {"arm": "right", "name": "pick_place", "prompt": "put block in bowl", "success": "block_in_bowl"}},
        {"check": "block_in_bowl"}]}})

    class FakeModel:
        def __init__(self):
            self.prompts = []

        def complete(self, system, user, images=()):
            self.prompts.append(user)
            return "```yaml\ntree: {skill: {arm: middle, name: pick_place}}\n```" if len(self.prompts) == 1 else good

    m = FakeModel()
    planner = LLMPlanner(m, arms=["left", "right"], skills={"pick_place": "", "home": ""}, predicates=[])
    spec = planner.plan("put the block in the bowl")
    assert spec.goal == "block_in_bowl"
    assert "unknown arm" in m.prompts[1]
