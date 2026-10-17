"""Run a TaskSpec end to end and report a verified outcome."""

from __future__ import annotations

import time
from dataclasses import dataclass

from liminal.core.types import Status
from liminal.orchestration.context import ExecutionContext
from liminal.orchestration.task_spec import TaskSpec
from liminal.safety.supervisor import ArmConflict, EmergencyStop


@dataclass
class RunReport:
    task: str
    status: Status
    goal_verified: bool
    duration_s: float
    error: str | None = None


async def run_task(task: TaskSpec, ctx: ExecutionContext) -> RunReport:
    ctx.world.update(task.initial_state, source="task_init")
    ctx.world.log("task_start", task=task.name)
    t0 = time.monotonic()
    error = None
    try:
        status = await task.tree.run(ctx)
    except (ArmConflict, EmergencyStop) as e:
        ctx.safety.estop()
        status, error = Status.FAILURE, f"{type(e).__name__}: {e}"
    # Never report success on the tree's word alone: re-check the goal predicate.
    goal_ok = status.ok and (task.goal is None or ctx.world.holds(task.goal))
    report = RunReport(task.name, status if goal_ok or not status.ok else Status.FAILURE,
                       goal_ok, time.monotonic() - t0, error)
    ctx.world.log("task_end", task=task.name, status=report.status.value, goal_verified=goal_ok, error=error)
    return report
