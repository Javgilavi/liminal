"""Assembly sequencing checks use mock arms and no optional simulation/ML imports."""

import asyncio
import importlib.util
from pathlib import Path

from liminal.core.types import SkillResult, Status
from liminal.orchestration.executor import run_task

path = Path(__file__).resolve().parents[1] / "scripts/sim_flux_assembly/harness.py"
spec = importlib.util.spec_from_file_location("assembly_harness", path)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class Backend:
    def __init__(self, fail=None, lie=None, lose_gear=False):
        self.calls, self.fail, self.lie, self.lose_gear = [], fail, lie, lose_gear

    async def execute(self, skill, ctx):
        self.calls.append(skill.name)
        assert ctx.safety.owner_of(skill.arm) is not None
        if skill.name == self.fail:
            return SkillResult(Status.FAILURE, skill)
        if skill.name != self.lie:
            ctx.world.set(skill.success, True, source="mock_detector")
        if self.lose_gear and skill.name == "pick_collar":
            ctx.world.set("gear_seated", False, source="mock_detector")
        return SkillResult(Status.SUCCESS, skill)


def test_assembly_runs_atomic_skills_in_order(make_ctx):
    backend = Backend()
    ctx = make_ctx(backend)
    result = asyncio.run(run_task(harness.assembly_task("right"), ctx))
    assert result.status is Status.SUCCESS and result.goal_verified
    assert backend.calls == [row[0] for row in harness.SKILLS]


def test_assembly_stops_before_later_skills_on_failure(make_ctx):
    backend = Backend(fail="seat_gear")
    ctx = make_ctx(backend)
    result = asyncio.run(run_task(harness.assembly_task("right"), ctx))
    assert result.status is Status.FAILURE and not result.goal_verified
    assert backend.calls == ["pick_gear", "seat_gear"]


def test_assembly_rejects_unverified_skill_success(make_ctx):
    backend = Backend(lie="seat_gear")
    ctx = make_ctx(backend)
    result = asyncio.run(run_task(harness.assembly_task("right"), ctx))
    assert result.status is Status.FAILURE
    assert backend.calls == ["pick_gear", "seat_gear"]


def test_assembly_rechecks_prior_part_before_final_insertion(make_ctx):
    backend = Backend(lose_gear=True)
    ctx = make_ctx(backend)
    result = asyncio.run(run_task(harness.assembly_task("right"), ctx))
    assert result.status is Status.FAILURE
    assert backend.calls == ["pick_gear", "seat_gear", "pick_collar"]
