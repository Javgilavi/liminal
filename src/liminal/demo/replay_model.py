"""Offline, deterministic demo model: answers both stages from the demo's ``annotation.yaml``.

This is SCRIPTED. No vision model looks at the frames; ``annotation.yaml`` is the ground truth of what the
human did, written by whoever authored the fixture. Results from it prove the pipeline (validation, binding,
compilation, execution), not that a model can read video. ``label``/``scripted`` flow into ``DemoExtraction``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from liminal.demo.extract import DESCRIBE_SYSTEM, MAP_SYSTEM
from liminal.demo.video import Demo, DemoError


class ReplayDemoModel:
    scripted = True
    label = "scripted replay model (reads annotation.yaml; no vision model involved)"

    def __init__(self, demo: Demo):
        if not demo.annotation:
            raise DemoError(f"demo {demo.name!r} has no annotation.yaml for the scripted model")
        self.demo = demo
        self.annotation: dict[str, Any] = demo.annotation
        self.calls: list[str] = []

    def _frames(self, refs: Any) -> list[str]:
        out: list[str] = []
        for ref in refs or []:
            full = self.demo.nearest_frame(ref)  # a sparser sampling cites the closest loaded frame
            if full not in out:
                out.append(full)
        return out

    def complete(self, system: str, user: str, images: Sequence[bytes] = ()) -> str:
        self.calls.append("describe" if system == DESCRIBE_SYSTEM else "map")
        if system == DESCRIBE_SYSTEM:
            return json.dumps(self._describe())
        if system == MAP_SYSTEM:
            return json.dumps(self._map(json.loads(user.split("\nYOUR PREVIOUS REPLY", 1)[0])))
        raise DemoError("ReplayDemoModel only answers the demo describe/map prompts")

    def _describe(self) -> dict[str, Any]:
        a = self.annotation
        subtasks = []
        for s in a.get("subtasks", []):
            keep = ("id", "action", "object", "destination", "confidence", "uncertain", "hidden", "transcript", "note")
            subtasks.append({**{k: s[k] for k in keep if k in s}, "frames": self._frames(s.get("frames"))})
        return {"object_changes": [{**c, "frames": self._frames(c.get("frames"))} for c in a.get("object_changes", [])],
                "subtasks": subtasks,
                "constraints": [{**c, "frames": self._frames(c.get("frames"))} for c in a.get("constraints", [])]}

    def _map(self, request: dict[str, Any]) -> dict[str, Any]:
        skills = {s["skill"]: s for s in request["skills"]}
        skill_of = {s["id"]: s.get("skill") for s in self.annotation.get("subtasks", [])}
        arm_of = {s["id"]: s.get("arm") for s in self.annotation.get("subtasks", [])}
        bindings = request["bindings"]
        steps: list[dict[str, Any]] = []
        unsupported: list[str] = []
        for sub in request["description"]["subtasks"]:
            sid, skill = sub["id"], skills.get(skill_of.get(sub["id"]) or "")
            if sub.get("hidden"):
                continue  # reported by trusted code
            if skill is None:
                unsupported.append(f"{sid}: {sub['action']} {sub['object']} has no registered skill")
                continue
            obj, dest = bindings.get(sub["object"]), bindings.get(sub.get("destination") or "") or ""
            if obj is None or (sub.get("destination") and not dest):
                continue  # unbound role: left out, never guessed
            steps.append({
                "id": sid, "skill": skill["skill"], "object_id": obj, "destination_id": dest, "arm": arm_of.get(sid),
                "effect": skill["effect_template"].format(object_id=obj, destination_id=dest),
                "depends_on": [c["before"] for c in request["description"]["constraints"] if c["after"] == sid],
                "confidence": sub.get("confidence"),
                "sources": [{"kind": "demo_frame", "ref": f} for f in sub["frames"]]})
        kept = {s["id"] for s in steps}
        for s in steps:
            s["depends_on"] = [d for d in s["depends_on"] if d in kept]
        return {"steps": steps, "obligations": [s["effect"] for s in steps], "unsupported": unsupported,
                "unresolved": []}
