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
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence as Seq
from contextlib import AsyncExitStack

from liminal.core.types import Skill, Status
from liminal.orchestration.attempt import AttemptScope
from liminal.orchestration.context import ExecutionContext
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
        # Set by a While that owns this node as its establish/release step: the arm is already
        # claimed for the whole hold lifetime, so this node adopts that claim instead of
        # re-claiming (which would be an ArmConflict) or releasing it (which would open a gap).
        self.claimed_by: str | None = None

    async def _run(self, ctx: ExecutionContext) -> Status:
        if ctx.attempts is not None and self.skill.success:
            if ctx.verification is not None:
                raise ValueError("a context carries either one attempt scope or an attempt factory, not both")
            # Open (and identify) the attempt before dispatch so its evidence window
            # cannot include anything observed for an earlier try.
            async with ctx.attempts.open(self.skill, self.name) as scope:
                return await self._execute(ctx, scope)
        return await self._execute(ctx, ctx.verification)

    async def _execute(self, ctx: ExecutionContext, scope: AttemptScope | None) -> Status:
        s = self.skill
        ctx.safety.check()
        members = ctx.members(s.arm)  # one arm, or every arm of a bimanual group
        async with AsyncExitStack() as stack:
            for arm in members:
                if self.claimed_by is not None and ctx.safety.owner_of(arm) == self.claimed_by:
                    continue
                await stack.enter_async_context(ctx.safety.claim(arm, owner=self.name))
            await stack.enter_async_context(ctx.safety.zones(s.zones))
            try:
                result = await asyncio.wait_for(ctx.skills.execute(s, ctx), timeout=s.timeout_s)
            except asyncio.TimeoutError:
                for arm in members:
                    ctx.arms[arm].stop()
                ctx.world.log("skill_timeout", node=self.name)
                return Status.TIMEOUT
            except asyncio.CancelledError:
                for arm in members:
                    ctx.arms[arm].stop()
                ctx.world.log("skill_cancelled", node=self.name)
                raise
            ctx.world.log("skill_result", node=self.name, status=result.status.value, message=result.message)
            if not result.status.ok:
                return result.status
            if scope is not None:
                from liminal.perception.evidence import Truth
                from liminal.perception.verifier import VerificationRequest

                if not s.success or s.success != scope.expected_effect:
                    ctx.world.log("verify_failed", node=self.name, reason="attempt effect mismatch")
                    return Status.FAILURE
                # Establish the post-execution boundary BEFORE obtaining the next
                # observation. A late delivery of a pre-action frame cannot pass.
                boundary = time.monotonic()
                sequence = ctx.world.sequence
                await scope.refresh()
                scope.result = await scope.verifier.verify(VerificationRequest(
                    run_id=scope.run_id, attempt_id=scope.attempt_id,
                    expected_effect=scope.expected_effect, object_bindings=scope.bindings,
                    evidence_after_seq=sequence, allowed_observers=scope.allowed_observers,
                    deadline_mono_s=time.monotonic() + s.verify_timeout_s,
                    evidence_after_mono_s=boundary,
                ), ctx.world)
                ctx.world.log("verification", node=self.name, decision=scope.result.decision.value,
                              evidence_ids=scope.result.evidence_ids, reason=scope.result.reason)
                return Status.SUCCESS if scope.result.decision is Truth.TRUE else Status.FAILURE
            if s.success and not await ctx.world.wait_for(s.success, timeout=s.verify_timeout_s):
                ctx.world.log("verify_failed", node=self.name, condition=s.success)
                return Status.FAILURE
            return Status.SUCCESS


class Check(Node):
    def __init__(self, condition: str, name: str | None = None):
        super().__init__(name or f"check:{condition}")
        self.condition = condition

    async def _run(self, ctx: ExecutionContext) -> Status:
        if ctx.observe is not None:
            await ctx.observe()  # judge the current scene, not the last thing we happened to see
        return Status.SUCCESS if ctx.world.holds(self.condition) else Status.FAILURE


class WaitFor(Node):
    def __init__(self, condition: str, timeout_s: float | None = 30.0, name: str | None = None):
        super().__init__(name or f"wait:{condition}")
        self.condition, self.timeout_s = condition, timeout_s

    async def _run(self, ctx: ExecutionContext) -> Status:
        ok = await ctx.world.wait_for(self.condition, timeout=self.timeout_s)
        return Status.SUCCESS if ok else Status.TIMEOUT


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
    """WHILE / UNTIL: `support_arm` maintains `maintain` while `body` runs.

    Lifecycle, all inside ONE claim of the support arm (every member of a group):

    1. `establish` (optional SkillNode on the support arm) creates the condition, in its own
       evidence attempt. Without it the condition must already hold (e.g. an earlier skill).
    2. the arm switches to the local position-hold controller (`hold_mode` is bookkeeping)
    3. `maintain` must hold from FRESH evidence (a refresh is forced in evidence mode)
    4. `body` runs on another arm in its own attempt, while `maintain` is monitored: in
       evidence mode `ctx.observe` is called every `monitor_period_s`, because nothing else
       would look. Unknown, an expired observation or a failing observer all count as loss.
       Loss cancels the body (its arms are stopped) and FAILS the node so an enclosing
       Retry/Fallback can recover. The hold is dropped on every failure path.
    5. on body success the condition and `until` are re-observed AFTER the body returned, so
       a simultaneous completion can never hide a violation. If the condition was lost the
       body's attempt is voided (a partial two-arm commit is not a success).
    6. `release` (optional SkillNode on the support arm) runs last, still inside the claim.
       Without it the hold stays engaged after the node returns (legacy: release it with a
       later `release` skill; the arm is then unclaimed in between).

    WorldState has one active attempt at a time. The hold is deliberately NOT an attempt: it
    is a node-scoped lifetime whose evidence is the monitor's observations (observation-only
    ids, or the body's attempt id while the body runs) plus the `while_*` trace events.
    """

    def __init__(self, support_arm: str, maintain: str, body: Node, until: str | None = None,
                 establish_timeout_s: float = 5.0, name: str | None = None, *,
                 establish: Node | None = None, release: Node | None = None,
                 monitor_period_s: float = 0.1):
        super().__init__(name or f"while:{support_arm}:{maintain}")
        if monitor_period_s <= 0:
            raise ValueError("monitor_period_s must be positive")
        for role, node in (("establish", establish), ("release", release)):
            if node is None:
                continue
            if not isinstance(node, SkillNode) or node.skill.arm != support_arm:
                raise ValueError(f"While {role} must be a SkillNode on the support arm {support_arm!r}")
            node.claimed_by = self.name
        self.support_arm = support_arm
        self.maintain = maintain
        self.body = body
        self.until = until
        self.establish_timeout_s = establish_timeout_s
        self.establish = establish
        self.release = release
        self.monitor_period_s = monitor_period_s

    async def _run(self, ctx: ExecutionContext) -> Status:
        members = ctx.members(self.support_arm)
        keep_hold = False
        async with AsyncExitStack() as stack:
            for arm in members:
                await stack.enter_async_context(ctx.safety.claim(arm, owner=self.name))
            try:
                status, keep_hold = await self._cycle(ctx, members)
                return status
            finally:
                if not keep_hold:
                    self._drop_hold(ctx, members)

    async def _cycle(self, ctx: ExecutionContext, members: list[str]) -> tuple[Status, bool]:
        evidence = ctx.observe is not None
        if self.establish is not None:
            if not (await self.establish.run(ctx)).ok:
                ctx.world.log("while_not_established", node=self.name, condition=self.maintain, reason="establish")
                return Status.FAILURE, False
        for arm in members:
            ctx.arms[arm].hold()
            ctx.world.set(f"execution.{arm}.hold_mode", True, source=self.name)
        ctx.world.log("hold_engaged", node=self.name, arms=members, condition=self.maintain)
        if not await self._established(ctx, evidence):
            ctx.world.log("while_not_established", node=self.name, condition=self.maintain, reason="evidence")
            return Status.FAILURE, False

        history = getattr(ctx.attempts, "history", None)
        first_body_attempt = len(history) if history is not None else 0
        body = asyncio.create_task(self.body.run(ctx))
        monitor = asyncio.create_task(self._monitor(ctx, evidence))
        try:
            done, _ = await asyncio.wait({body, monitor}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (body, monitor):
                t.cancel()
            await asyncio.gather(body, monitor, return_exceptions=True)

        body_status = None if body.cancelled() else body.result()  # re-raises ArmConflict & co.
        lost = monitor in done
        if body_status is not None and evidence:
            # The body may have finished in the same instant the condition broke: judge both
            # from an observation taken after it returned, never from the last one we saw.
            lost = not await self._refresh(ctx) or lost
        lost = lost or not ctx.world.holds(self.maintain)
        if lost:
            ctx.world.log("while_violated", node=self.name, condition=self.maintain,
                          body_status=body_status.value if body_status else None,
                          evidence_ids=self._evidence_ids(ctx))
            if body_status is not None and body_status.ok:
                void = getattr(ctx.attempts, "void_since", None)
                if void is not None:
                    void(first_body_attempt, f"{self.maintain} lost while the body ran")
            return Status.FAILURE, False
        assert body_status is not None  # the monitor is the only other way out
        if not body_status.ok:
            return body_status, False
        if self.until and not ctx.world.holds(self.until):
            ctx.world.log("while_until_unmet", node=self.name, condition=self.until)
            return Status.FAILURE, False
        if self.release is None:
            return Status.SUCCESS, True
        released = await self.release.run(ctx)
        ctx.world.log("hold_released", node=self.name, arms=members, status=released.value)
        return (Status.SUCCESS if released.ok else released), False

    async def _established(self, ctx: ExecutionContext, evidence: bool) -> bool:
        if not evidence:
            return await ctx.world.wait_for(self.maintain, timeout=self.establish_timeout_s)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.establish_timeout_s
        while True:
            if await self._refresh(ctx) and ctx.world.holds(self.maintain):
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(self.monitor_period_s, remaining))

    async def _refresh(self, ctx: ExecutionContext) -> bool:
        """One fresh observation. A failing observer is a loss of sight, not a pass."""
        try:
            await ctx.observe()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any observer failure means we cannot see
            ctx.world.log("while_observe_failed", node=self.name, error=f"{type(exc).__name__}: {exc}")
            return False
        return True

    async def _monitor(self, ctx: ExecutionContext, evidence: bool) -> None:
        """Return when `maintain` is false, unknown or expired, or when we can no longer look."""
        tasks = [asyncio.create_task(ctx.world.wait_until_violated(self.maintain))]
        if evidence:
            tasks.append(asyncio.create_task(self._watch(ctx)))
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _watch(self, ctx: ExecutionContext) -> None:
        while await self._refresh(ctx) and ctx.world.holds(self.maintain):
            await asyncio.sleep(self.monitor_period_s)

    def _evidence_ids(self, ctx: ExecutionContext) -> list[str]:
        records = ctx.world.evidence_records(self.maintain)
        return list(records[-1].evidence_ids) if records else []

    def _drop_hold(self, ctx: ExecutionContext, members: list[str]) -> None:
        # Never leave a hold controller running on an arm nobody owns any more.
        for arm in members:
            ctx.arms[arm].release_hold()
            ctx.world.set(f"execution.{arm}.hold_mode", False, source=self.name)


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
