"""Async behavior tree with temporal dependencies (owned by WS3).

Dependency types and how they map to nodes:

    AFTER  A then B                    -> Sequence([A, B])
    WHILE  A must stay true during B   -> While(support=..., maintain=cond, body=B)
    UNTIL  A continues until cond      -> While(..., until=cond)  /  WaitFor(cond)

Every node is a coroutine `run(ctx) -> Status`. Cancellation (asyncio.CancelledError)
is how a parent aborts a child, e.g. when a WHILE invariant breaks mid-body.
The tree enforces correctness; any LLM above it only *proposes* trees.
"""

from __future__ import annotations

import asyncio
import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence as Seq
from contextlib import AsyncExitStack

from liminal.core.types import Skill, Status
from liminal.orchestration.context import ExecutionContext
from liminal.perception.types import InspectionRequest
from liminal.safety.supervisor import ArmConflict, EmergencyStop

_ids = itertools.count()


class Node(ABC):
    def __init__(self, name: str | None = None):
        self.name = name or f"{type(self).__name__.lower()}_{next(_ids)}"

    async def run(self, ctx: ExecutionContext) -> Status:
        ctx.world.log("node_start", node=self.name, type=type(self).__name__)
        try:
            status = await self._run(ctx)
        except asyncio.CancelledError:
            ctx.world.log("node_end", node=self.name, status=Status.CANCELLED.value)
            raise
        except (ArmConflict, EmergencyStop):
            ctx.world.log("node_end", node=self.name, status=Status.FAILURE.value, error="safety")
            raise
        ctx.world.log("node_end", node=self.name, status=status.value)
        return status

    @abstractmethod
    async def _run(self, ctx: ExecutionContext) -> Status: ...


# -- leaves --------------------------------------------------------------------------


class SkillNode(Node):
    """Run one skill on one arm, then independently verify its success predicate."""

    def __init__(self, skill: Skill, name: str | None = None):
        super().__init__(name or skill.label)
        self.skill = skill

    async def _run(self, ctx: ExecutionContext) -> Status:
        s = self.skill
        ctx.safety.check()
        members = ctx.members(s.arm)  # one arm, or every arm of a bimanual group
        async with AsyncExitStack() as stack:
            for arm in members:
                await stack.enter_async_context(ctx.safety.claim(arm, owner=self.name))
            await stack.enter_async_context(ctx.safety.zones(s.zones))
            try:
                result = await asyncio.wait_for(ctx.skills.execute(s, ctx), timeout=s.timeout_s)
            except asyncio.TimeoutError:
                for arm in members:
                    ctx.arms[arm].stop()
                ctx.world.log("skill_timeout", node=self.name)
                return Status.TIMEOUT
        ctx.world.log("skill_result", node=self.name, status=result.status.value, message=result.message)
        if not result.status.ok:
            return result.status
        if s.success and not await ctx.world.wait_for(s.success, timeout=s.verify_timeout_s):
            ctx.world.log("verify_failed", node=self.name, condition=s.success)
            return Status.FAILURE
        return Status.SUCCESS


class Check(Node):
    def __init__(self, condition: str, name: str | None = None):
        super().__init__(name or f"check:{condition}")
        self.condition = condition

    async def _run(self, ctx: ExecutionContext) -> Status:
        return Status.SUCCESS if ctx.world.holds(self.condition) else Status.FAILURE


class WaitFor(Node):
    def __init__(self, condition: str, timeout_s: float | None = 30.0, name: str | None = None):
        super().__init__(name or f"wait:{condition}")
        self.condition, self.timeout_s = condition, timeout_s

    async def _run(self, ctx: ExecutionContext) -> Status:
        ok = await ctx.world.wait_for(self.condition, timeout=self.timeout_s)
        return Status.SUCCESS if ok else Status.TIMEOUT


class Inspect(Node):
    """Inspect until every requested predicate is known or the deadline expires."""

    def __init__(
        self,
        request: InspectionRequest,
        timeout_s: float | None = None,
        retry_delay_s: float = 0.05,
        name: str | None = None,
    ):
        super().__init__(name or f"inspect:{','.join(request.predicates)}")
        self.request = request
        self.timeout_s = request.timeout_s if timeout_s is None else timeout_s
        self.retry_delay_s = retry_delay_s
        if self.timeout_s <= 0 or retry_delay_s < 0:
            raise ValueError("inspection timeout must be positive and retry delay non-negative")

    async def _run(self, ctx: ExecutionContext) -> Status:
        if ctx.inspector is None:
            ctx.world.log("inspect_unavailable", node=self.name)
            return Status.FAILURE

        assessment = await ctx.inspector.inspect_until_known(
            self.request,
            timeout_s=self.timeout_s,
            retry_delay_s=self.retry_delay_s,
        )
        values = {key: observation.value.value for key, observation in assessment.observations.items()}
        ctx.world.log("inspection", node=self.name, values=values, parse_error=assessment.parse_error)
        if all(assessment.observation(predicate).known for predicate in self.request.predicates):
            return Status.SUCCESS
        ctx.world.log("inspect_timeout", node=self.name, predicates=self.request.predicates)
        return Status.TIMEOUT


class SetFact(Node):
    """Write a fact into world state (bookkeeping, e.g. subgoal_done markers)."""

    def __init__(self, key: str, value: object = True, name: str | None = None):
        super().__init__(name or f"set:{key}")
        self.key, self.value = key, value

    async def _run(self, ctx: ExecutionContext) -> Status:
        ctx.world.set(self.key, self.value, source=self.name)
        return Status.SUCCESS


# -- composites ----------------------------------------------------------------------


class Sequence(Node):
    """AFTER: run children in order, stop at the first failure."""

    def __init__(self, children: Seq[Node], name: str | None = None):
        super().__init__(name)
        self.children = list(children)

    async def _run(self, ctx: ExecutionContext) -> Status:
        for child in self.children:
            status = await child.run(ctx)
            if not status.ok:
                return status
        return Status.SUCCESS


class Fallback(Node):
    """Try children in order until one succeeds (recovery alternatives)."""

    def __init__(self, children: Seq[Node], name: str | None = None):
        super().__init__(name)
        self.children = list(children)

    async def _run(self, ctx: ExecutionContext) -> Status:
        status = Status.FAILURE
        for child in self.children:
            status = await child.run(ctx)
            if status.ok:
                return status
        return status


class Parallel(Node):
    """Run children concurrently. Succeeds when all succeed; the first failure cancels
    the rest. Children on the *same* arm will raise ArmConflict - that is intended."""

    def __init__(self, children: Seq[Node], name: str | None = None):
        super().__init__(name)
        self.children = list(children)

    async def _run(self, ctx: ExecutionContext) -> Status:
        tasks = [asyncio.create_task(c.run(ctx)) for c in self.children]
        try:
            for fut in asyncio.as_completed(tasks):
                status = await fut
                if not status.ok:
                    return status
            return Status.SUCCESS
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class While(Node):
    """WHILE / UNTIL: `support_arm` holds its pose while `body` runs.

    1. claim `support_arm` and switch it to the local position-hold controller
    2. wait until `maintain` holds (e.g. "lid_open")
    3. run `body`, monitoring `maintain` concurrently; if it breaks, cancel the body
       and FAIL (so an enclosing Retry/Fallback can recover)
    4. on body success, require `until` (e.g. "object_inserted") if given

    The hold stays engaged after this node returns; release it explicitly with a
    `release` skill so the release is visible and ordered in the tree.
    """

    def __init__(self, support_arm: str, maintain: str, body: Node, until: str | None = None,
                 establish_timeout_s: float = 5.0, name: str | None = None):
        super().__init__(name or f"while:{support_arm}:{maintain}")
        self.support_arm = support_arm
        self.maintain = maintain
        self.body = body
        self.until = until
        self.establish_timeout_s = establish_timeout_s

    async def _run(self, ctx: ExecutionContext) -> Status:
        async with AsyncExitStack() as stack:
            for arm in ctx.members(self.support_arm):
                await stack.enter_async_context(ctx.safety.claim(arm, owner=self.name))
                ctx.arms[arm].hold()
                ctx.world.set(f"{arm}_holding", True, source=self.name)
            if not await ctx.world.wait_for(self.maintain, timeout=self.establish_timeout_s):
                ctx.world.log("while_not_established", node=self.name, condition=self.maintain)
                return Status.FAILURE

            with ctx.watch_invariant(self.maintain):
                body = asyncio.create_task(self.body.run(ctx))
                monitor = asyncio.create_task(ctx.world.wait_until_violated(self.maintain))
                try:
                    done, _ = await asyncio.wait({body, monitor}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for t in (body, monitor):
                        t.cancel()
                    await asyncio.gather(body, monitor, return_exceptions=True)

            if monitor in done and not ctx.world.holds(self.maintain):
                ctx.world.log("while_violated", node=self.name, condition=self.maintain)
                return Status.FAILURE
            status = body.result()
            if status.ok and self.until and not ctx.world.holds(self.until):
                return Status.FAILURE
            return status


# -- decorators ----------------------------------------------------------------------


class Retry(Node):
    """Retry `child` up to `attempts` times, running `recovery` between attempts."""

    def __init__(self, child: Node, attempts: int = 2, recovery: Node | None = None, name: str | None = None):
        super().__init__(name or f"retry:{child.name}")
        self.child, self.attempts, self.recovery = child, attempts, recovery

    async def _run(self, ctx: ExecutionContext) -> Status:
        status = Status.FAILURE
        for i in range(self.attempts):
            status = await self.child.run(ctx)
            if status.ok:
                return status
            ctx.world.log("retry", node=self.name, attempt=i + 1, status=status.value)
            if self.recovery is not None and i + 1 < self.attempts:
                await self.recovery.run(ctx)
        return status


class Timeout(Node):
    def __init__(self, child: Node, seconds: float, name: str | None = None):
        super().__init__(name or f"timeout:{child.name}")
        self.child, self.seconds = child, seconds

    async def _run(self, ctx: ExecutionContext) -> Status:
        try:
            return await asyncio.wait_for(self.child.run(ctx), timeout=self.seconds)
        except asyncio.TimeoutError:
            return Status.TIMEOUT
