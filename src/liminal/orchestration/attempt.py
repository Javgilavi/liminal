"""Runtime-owned verification scope for one bounded skill attempt."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

from liminal.perception.evidence import Bindings, Truth
from liminal.perception.verifier import VerificationResult, Verifier

if TYPE_CHECKING:
    from liminal.core.types import Skill
    from liminal.perception.world_state import WorldState


@dataclass
class AttemptScope:
    run_id: str
    attempt_id: str
    expected_effect: str
    bindings: Bindings
    allowed_observers: tuple[str, ...]
    refresh: Callable[[], Awaitable[None]]
    verifier: Verifier
    result: VerificationResult | None = None
    skill: str = ""  # for reports only; the effect is what gets verified
    arm: str = ""
    # Set when an enclosing node later proved the verified effect did not stand on its own
    # (e.g. a While lost its maintained condition while the body ran).
    voided: str | None = None


class AttemptFactory(Protocol):
    """Opens a fresh attempt around each verified skill execution in an authored tree."""

    def open(self, skill: Skill, node: str) -> AbstractAsyncContextManager[AttemptScope]: ...


def _no_bindings(_: Skill) -> Bindings:
    return ()


@dataclass
class EvidenceAttempts:
    """V1 attempts for authored procedures: one new attempt identity per skill run.

    A Retry iteration therefore gets its own attempt id, so evidence gathered for a
    failed try can never verify the next one. The caller owns `world.begin_run`.
    """

    world: WorldState
    run_id: str
    refresh: Callable[[], Awaitable[None]]
    allowed_observers: tuple[str, ...]
    verifier: Verifier = field(default_factory=Verifier)
    bindings_for: Callable[[Skill], Bindings] = _no_bindings
    history: list[AttemptScope] = field(default_factory=list)

    @asynccontextmanager
    async def open(self, skill: Skill, node: str) -> AsyncIterator[AttemptScope]:
        if not skill.success:
            raise ValueError(f"{node}: an evidence-backed attempt needs a success predicate")
        scope = AttemptScope(
            run_id=self.run_id, attempt_id=f"{self.run_id}:{node}:{uuid.uuid4().hex[:8]}",
            expected_effect=skill.success, bindings=self.bindings_for(skill),
            allowed_observers=self.allowed_observers, refresh=self.refresh, verifier=self.verifier,
            skill=skill.name, arm=skill.arm,
        )
        self.world.begin_attempt(scope.attempt_id)
        self.world.log("attempt_start", node=node, attempt_id=scope.attempt_id, effect=scope.expected_effect)
        try:
            yield scope
        finally:
            self.world.end_attempt()
            decision = scope.result.decision.value if scope.result else None
            self.world.log("attempt_end", node=node, attempt_id=scope.attempt_id, decision=decision)
            self.history.append(scope)

    def void_since(self, index: int, reason: str) -> list[str]:
        """Turn verified attempts recorded from `history[index:]` into failures.

        A cooperative block (support arm + body arm) commits as a whole: when the support
        was lost while the body ran, the body's own verification no longer counts, so the
        report and the ledger never show a "verified" insert inside a failed block.
        """
        voided = []
        for scope in self.history[index:]:
            if scope.result is not None and scope.result.decision is Truth.TRUE:
                scope.result = replace(scope.result, decision=Truth.FALSE, reason=f"voided: {reason}")
                scope.voided = reason
                voided.append(scope.attempt_id)
                self.world.log("attempt_voided", attempt_id=scope.attempt_id, reason=reason)
        return voided
