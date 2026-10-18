"""Embodiment descriptions and action-space mapping (owned by WS4).

The current runtime uses task LoRA and two independent 6-D per-arm policies.
The original unified XLeRobot fine-tune is superseded. Historical mapping modes
remain compatible, as described in docs/research/04_xlerobot_sim.md:

  A. "per_arm"  - one policy session per arm, each seeing its own 6-D slice + wrist
     camera; the orchestrator coordinates. No retraining of the action head.
  B. "bimanual" - fine-tune on a concatenated action vector (e.g. 12-D for two arms,
     + head/base for XLeRobot). Needs a bimanual dataset and an adapted action head.

`Embodiment.split` / `merge` let both modes share the same dataset conversion code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ActionGroup:
    name: str  # e.g. "left_arm", "right_arm", "head", "base"
    joints: list[str]
    policy_controlled: bool = True


@dataclass
class Embodiment:
    name: str
    groups: list[ActionGroup]
    cameras: dict[str, str] = field(default_factory=dict)  # role -> camera name
    mode: str = "per_arm"

    @classmethod
    def load(cls, path: str | Path) -> Embodiment:
        d = yaml.safe_load(Path(path).read_text())
        groups = [ActionGroup(**g) for g in d.pop("groups")]
        return cls(groups=groups, **d)

    @property
    def action_dim(self) -> int:
        return sum(len(g.joints) for g in self.groups if g.policy_controlled)

    def slices(self) -> dict[str, slice]:
        out, i = {}, 0
        for g in self.groups:
            if g.policy_controlled:
                out[g.name] = slice(i, i + len(g.joints))
                i += len(g.joints)
        return out

    def split(self, action: list[float]) -> dict[str, list[float]]:
        if len(action) != self.action_dim:
            raise ValueError(f"{self.name}: expected {self.action_dim}-D action, got {len(action)}")
        return {k: list(action[s]) for k, s in self.slices().items()}

    def merge(self, parts: dict[str, list[float]]) -> list[float]:
        out: list[float] = []
        for g in self.groups:
            if g.policy_controlled:
                out.extend(parts[g.name])
        return out
