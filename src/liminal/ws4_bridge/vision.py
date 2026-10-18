"""Scene predicates from ws4's measured Claude Haiku success checks (provider ``ws4_haiku``).

Each predicate maps to the ws4 skill whose check decides it, with that skill's tuned scene prompt, hint and condition
(ws4/orchestrator/skills.yaml; held-out accuracy on recorded Haller head frames: lid 63/63, green pen 114/119,
cutter 91/91, red pen 65/66, research/orchestrator.md). The checks were tuned on the head camera only, so only the
``scene_camera`` image is sent. A verdict below the skill's confidence threshold is ``unknown`` (the prompts ask for
low confidence when the object is hidden or ambiguous). A ``derived`` predicate is true when all its parts are.

``keep_s``: for a predicate an arm keeps true physically (the box arm frozen holding the lid), an unclear view is no
evidence either way: the last confident verdict is reported again while it is at most ``keep_s`` old, then the
predicate is unknown (fail closed). Without it, one occluded frame breaks a WHILE: on 785 head frames through 53 pen
takes the lid check never read "closed" but was unsure on 74 (the pen arm hiding the box), enough to break the WHILE
in 37 of the 53 takes; with 30 s or more, in none (configs/perception/haller_haiku.yaml).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from liminal.perception.types import InspectionRequest

# liminal predicate -> ws4 skill whose success check decides it
HALLER_CHECKS = {
    "lid_open": "open_box",
    "green_pen_in_slot": "put_green_pen_in_slot",
    "yellow_boxcutter_in_slot": "put_yellow_cutter_in_slot",
    "red_pen_in_slot": "put_pen_in_slot",
}
HALLER_DERIVED = {"components_assembled": ["green_pen_in_slot", "red_pen_in_slot", "yellow_boxcutter_in_slot"]}


class Ws4HaikuVisionModel:
    def __init__(self, checks: Mapping[str, str] | None = None, derived: Mapping[str, Sequence[str]] | None = None,
                 scene_camera: str = "top", vocabulary: str | None = None, budget_usd: float = 2.0,
                 effort: str = "low", thinking: str = "adaptive", keep_s: Mapping[str, float] | None = None,
                 verifier: Any = None, clock: Callable[[], float] = time.monotonic, verifier_model: str = ""):
        self.checks = dict(HALLER_CHECKS if checks is None else checks)
        self.derived = {k: list(v) for k, v in (HALLER_DERIVED if derived is None else derived).items()}
        for name, parts in self.derived.items():
            missing = set(parts) - set(self.checks)
            if missing or name in self.checks:
                raise ValueError(f"derived predicate {name!r} needs checked parts and no check of its own ({missing})")
        self.scene_camera = scene_camera
        self.vocabulary, self.budget_usd, self.effort, self.thinking = vocabulary, budget_usd, effort, thinking
        self.keep_s = {k: float(v) for k, v in (keep_s or {}).items()}
        if set(self.keep_s) - set(self.checks):
            raise ValueError(f"keep_s names unchecked predicates {sorted(set(self.keep_s) - set(self.checks))}")
        self._verifier = verifier
        self._vocab: Any = None
        self.clock = clock
        self.verifier_model = verifier_model  # "" = each ws4 skill's own (Haiku 5.5)
        self._confident: dict[str, tuple[str, float]] = {}  # predicate -> (last confident value, when)
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the ws4 vocabulary and the verifier; with no API key, fail now rather than inside a WHILE."""
        from liminal.ws4_bridge import REPO, import_orchestrator

        import_orchestrator()
        from orchestrator.claude import ClaudeJSON, api_key_available  # noqa: PLC0415
        from orchestrator.cost import CostMeter  # noqa: PLC0415
        from orchestrator.skills import load_vocabulary  # noqa: PLC0415
        from orchestrator.verifier import ClaudeVerifier  # noqa: PLC0415

        path = self.vocabulary
        if path and not str(path).startswith("/"):
            path = str(REPO / path)
        self._vocab = load_vocabulary(path)
        unknown = sorted(set(self.checks.values()) - set(self._vocab.skills))
        if unknown:
            raise ValueError(f"ws4 vocabulary has no checks for {unknown}")
        if self._verifier is None:
            if not api_key_available():
                raise RuntimeError("ws4_haiku perception needs an Anthropic API key (ANTHROPIC_API_KEY or the ws4 "
                                   "key file, see ws4/orchestrator/claude.py)")
            self._verifier = ClaudeVerifier(ClaudeJSON(CostMeter(self.budget_usd)), effort=self.effort,
                                            haiku_thinking=self.thinking, station=self._vocab.station)

    async def inspect(self, prompt: str, images: list[Any]) -> Mapping[str, Any]:
        raise RuntimeError("ws4_haiku checks one predicate at a time; SceneInspector calls inspect_request")

    async def inspect_request(self, request: InspectionRequest, images: list[Any]) -> Mapping[str, Any]:
        if self._vocab is None:
            await asyncio.to_thread(self.load)
        frames = dict(zip(request.cameras, images, strict=True))
        if self.scene_camera not in frames:
            return _all(request.predicates, "unknown", f"the checks need the {self.scene_camera!r} camera")
        import numpy as np

        head = np.asarray(frames[self.scene_camera])
        wanted = [p for p in request.predicates if p in self.checks]
        for name in request.predicates:
            wanted += [p for p in self.derived.get(name, ()) if p not in wanted]
        results = await asyncio.gather(*(asyncio.to_thread(self._check, p, head) for p in wanted))
        out = dict(zip(wanted, results, strict=True))
        predicates: dict[str, dict[str, Any]] = {}
        for name in request.predicates:
            if name in out:
                predicates[name] = out[name]
            elif name in self.derived:
                parts = [out[p]["value"] for p in self.derived[name]]
                value = "false" if "false" in parts else "true" if all(v == "true" for v in parts) else "unknown"
                predicates[name] = {"value": value, "evidence": ", ".join(
                    f"{p}={v}" for p, v in zip(self.derived[name], parts, strict=True))}
            else:
                predicates[name] = {"value": "unknown", "evidence": "no ws4 check for this predicate"}
        return {"predicates": predicates, "ambiguous": []}

    def _check(self, predicate: str, head: Any) -> dict[str, Any]:
        from orchestrator.plan import Step  # noqa: PLC0415

        skill = self._vocab.skills[self.checks[predicate]]
        if self.verifier_model:
            import dataclasses

            skill = dataclasses.replace(skill, verifier_model=self.verifier_model)
        step = Step(id=predicate, arm=skill.arms[0], skill=skill.name, success_check=skill.success_check)
        verdict = self._verifier.check(step, skill, {"observation.images.head": head})
        evidence = verdict.observation + (f" [{verdict.error}]" if verdict.error else "")
        if verdict.error or verdict.confidence < skill.confidence:
            value = "unknown"
        else:
            value = "true" if verdict.done else "false"
        now = self.clock()
        with self._lock:
            if value != "unknown":
                self._confident[predicate] = (value, now)
            elif predicate in self.keep_s and predicate in self._confident:
                kept, when = self._confident[predicate]
                if now - when <= self.keep_s[predicate]:
                    return {"value": kept, "evidence": f"view unclear ({verdict.confidence:.2f}: {evidence}); kept the "
                                                       f"confident verdict from {now - when:.0f} s ago"}
        return {"value": value, "evidence": evidence, "confidence": float(verdict.confidence)}


def _all(predicates: Sequence[str], value: str, evidence: str) -> dict[str, Any]:
    return {"predicates": {p: {"value": value, "evidence": evidence} for p in predicates}, "ambiguous": []}
