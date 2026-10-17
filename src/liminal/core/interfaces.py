"""Interface contracts between workstreams.

These Protocols are the agreed API from hour 2 onward. Any implementation (real
SO-101, MuJoCo, mock) must satisfy them so modules can be swapped without touching
the orchestrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from liminal.core.types import JointPositions, Skill, SkillResult

if TYPE_CHECKING:
    from liminal.orchestration.context import ExecutionContext


@runtime_checkable
class Arm(Protocol):
    """One physical or simulated arm. Owned by WS5 (hardware) / WS4 (sim)."""

    name: str
    num_joints: int

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def read_state(self) -> JointPositions: ...
    def send_action(self, positions: JointPositions) -> None: ...

    def hold(self, positions: JointPositions | None = None) -> None:
        """Lock the arm at `positions` (default: current pose) using the local
        position controller. Must keep holding without any policy inference."""
        ...

    def release_hold(self) -> None: ...
    def stop(self) -> None:
        """Immediate stop: freeze at the current pose. Called by the e-stop."""
        ...


@runtime_checkable
class Camera(Protocol):
    name: str

    def read(self) -> Any:
        """Return the latest HxWx3 uint8 frame."""
        ...


@runtime_checkable
class SkillBackend(Protocol):
    """Executes a Skill on hardware/sim. Owned by WS1 (FLUX) and WS5 (primitives)."""

    async def execute(self, skill: Skill, ctx: ExecutionContext) -> SkillResult: ...


@runtime_checkable
class PredicateDetector(Protocol):
    """Turns camera frames into symbolic predicates. Owned by WS3."""

    def detect(self, frames: dict[str, Any]) -> dict[str, bool]: ...
