"""The fixed skill vocabulary (``skills.yaml``): what the planner may choose and how each skill is run and checked."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import PACKAGE_DIR

DEFAULT_VOCABULARY = PACKAGE_DIR / "skills.yaml"
ARMS = ("left", "right")
KINDS = ("flux", "primitive")
PRIMITIVES = ("hold", "home", "open_gripper", "close_gripper")
VERIFIER_MODELS = ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5")


@dataclass(frozen=True)
class Skill:
    name: str
    kind: str
    arms: tuple[str, ...]
    prompt: str = ""
    description: str = ""
    success_check: str = ""
    timeout_s: float = 45.0
    verify_every_chunks: int = 2
    verifier_model: str = "claude-haiku-5-5"
    confidence: float = 0.7
    max_attempts: int = 2
    package: str | None = None
    hold_after: bool = False
    keywords: tuple[str, ...] = ()
    sim: dict | None = None
    sim_check: dict | None = None

    @property
    def is_flux(self) -> bool:
        return self.kind == "flux"

    def planner_view(self) -> dict[str, Any]:
        """What Claude sees: no sim internals, no package names."""
        view: dict[str, Any] = {"name": self.name, "kind": self.kind, "arms": list(self.arms),
                                "description": self.description}
        if self.is_flux:
            view.update(flux_prompt=self.prompt, default_success_check=self.success_check,
                        timeout_s=self.timeout_s)
        return view


@dataclass
class Vocabulary:
    skills: dict[str, Skill]
    version: int = 1
    source: str = ""
    raw: dict = field(default_factory=dict)

    def __getitem__(self, name: str) -> Skill:
        return self.skills[name]

    def __contains__(self, name: object) -> bool:
        return name in self.skills

    def names(self) -> list[str]:
        return list(self.skills)

    def flux_prompts(self) -> list[str]:
        return [s.prompt for s in self.skills.values() if s.is_flux]

    def planner_view(self) -> list[dict[str, Any]]:
        return [s.planner_view() for s in self.skills.values()]


def load_vocabulary(path: str | Path | None = None) -> Vocabulary:
    import yaml

    path = Path(path) if path else DEFAULT_VOCABULARY
    data = yaml.safe_load(path.read_text()) or {}
    defaults = dict(data.get("defaults") or {})
    skills: dict[str, Skill] = {}
    errors: list[str] = []
    for name, spec in (data.get("skills") or {}).items():
        spec = {**defaults, **(spec or {})}
        kind = spec.get("kind")
        arms = tuple(spec.get("arms") or ())
        if kind not in KINDS:
            errors.append(f"{name}: kind must be one of {KINDS}")
        if not arms or any(a not in ARMS for a in arms):
            errors.append(f"{name}: arms must be a non-empty subset of {ARMS}")
        if kind == "flux" and not str(spec.get("prompt") or "").strip():
            errors.append(f"{name}: a flux skill needs its exact FLUX prompt")
        if kind == "primitive" and name not in PRIMITIVES:
            errors.append(f"{name}: unknown primitive (have {PRIMITIVES})")
        if spec.get("verifier_model") not in VERIFIER_MODELS:
            errors.append(f"{name}: verifier_model must be one of {VERIFIER_MODELS}")
        skills[name] = Skill(
            name=name,
            kind=kind,
            arms=arms,
            prompt=str(spec.get("prompt") or ""),
            description=str(spec.get("description") or ""),
            success_check=str(spec.get("success_check") or ""),
            timeout_s=float(spec.get("timeout_s", 45)),
            verify_every_chunks=max(1, int(spec.get("verify_every_chunks", 2))),
            verifier_model=str(spec.get("verifier_model")),
            confidence=float(spec.get("confidence", 0.7)),
            max_attempts=int(spec.get("max_attempts", 2)),
            package=spec.get("package"),
            hold_after=bool(spec.get("hold_after", False)),
            keywords=tuple(str(k).lower() for k in spec.get("keywords") or ()),
            sim=spec.get("sim"),
            sim_check=spec.get("sim_check"),
        )
    prompts = [s.prompt for s in skills.values() if s.is_flux]
    if len(prompts) != len(set(prompts)):
        errors.append("two flux skills share one prompt")
    if "hold" not in skills:
        errors.append("the vocabulary must define the hold primitive")
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    return Vocabulary(skills, int(data.get("version", 1)), str(path), data)
