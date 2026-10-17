"""Two-stage demo -> ProcedureProposal extraction (after Video-to-BT, arXiv 2509.16611).

Stage 1 *describe*: the model sees the demo frames (and transcript) and reports object-role changes,
candidate subtasks in order, and ordering/support constraints, each citing frame ids, with hidden or
uncertain actions flagged. It is told nothing about the robot or the current scene.

Stage 2 *map*: the stage-1 description (text only, no demo images) is mapped onto the CURRENT
``SkillWorld``. Trusted code here, not the model, then

* binds symbolic roles ("the red block") to current object ids by label; a role that does not bind
  uniquely is reported in ``unresolved`` and its step is dropped, never guessed,
* never maps hidden actions, and caps the confidence of uncertain ones,
* cites the demo frames from the stage-1 description on every step (a step with none is rejected),
* turns order constraints into ``depends_on`` and support constraints into ``maintain``,
* reports every subtask that did not become a step (``unsupported`` / ``unresolved``),
* validates everything with ``parse_procedure`` and feeds errors back for ``max_repairs`` rounds.

Demo frames are historical: they never become evidence about the current scene.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from liminal.demo.video import Demo, DemoError
from liminal.orchestration.procedure import (
    MAX_STEPS,
    SCHEMA_VERSION,
    ProcedureError,
    ProcedureProposal,
    SkillWorld,
    TextModel,
    extract_json,
    parse_procedure,
)

MAX_TEXT = 240
_REPLY_ERRORS = (ProcedureError, KeyError, TypeError)  # malformed model replies must reach the repair loop
_STEP_KEYS = {"id", "skill", "object_id", "destination_id", "arm", "depends_on", "effect", "maintain", "sources",
              "confidence", "note"}
CONSTRAINT_KINDS = ("order", "support")
UNCERTAIN_CONFIDENCE_CAP = 0.6
DEFAULT_CONFIDENCE = 0.5

DESCRIBE_SYSTEM = """You watch keyframes of a human demonstrating a tabletop manipulation task, plus an optional
transcript of what they said. You know nothing about any robot or any other scene. Report only what the
frames and transcript show. Describe objects by plain role phrases ("the red block", "the white box"); never
invent an identity you cannot see, and say so in "note" when unsure. Report:
  object_changes: how each object's role/state changed, citing frame ids;
  subtasks: the human's actions in demonstrated order (id d1, d2, ...), one per atomic manipulation,
    each with the frames that show it;
  constraints: "order" (before -> after) or "support" (the `after` subtask needs `condition` to stay true
    while it runs, kept true by the `before` subtask).
Flag "hidden": true for an action you could not see (occluded, off-camera, inferred only from the
transcript) and "uncertain": true when you are not sure. Cite only frame ids from "frames".
Reply with ONE JSON object and nothing else, matching "response_shape"."""

MAP_SYSTEM = """You map a human demonstration onto what the robot can do NOW. You receive the stage-1
description of the demo, the robot's registered skills, the CURRENT scene's object ids, and "bindings" from the
demo's role phrases to current object ids (null = no unique match). Rules:
- Use ONLY the listed skills, object ids and arms. One step per subtask; the step id MUST be the subtask id.
- Use exactly the bound object ids from "bindings". If a role's binding is null, do NOT guess an object:
  leave the step out and describe the problem in "unresolved" (start the entry with the subtask id and a colon).
- A demonstrated action with no registered skill goes in "unsupported" ("d2: ..."); never invent a skill.
- Do not map subtasks flagged "hidden". Put them in "unresolved".
- Cite the subtask's demo frames in the step "sources" (kind "demo_frame", ref = the frame id).
Reply with ONE JSON object and nothing else, matching "response_shape"."""


# -- stage-1 types -------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectChange:
    object: str
    change: str
    frames: tuple[str, ...]


@dataclass(frozen=True)
class Subtask:
    id: str
    action: str
    object: str
    destination: str
    frames: tuple[str, ...]
    confidence: float | None = None
    uncertain: bool = False
    hidden: bool = False
    transcript: str = ""
    note: str = ""


@dataclass(frozen=True)
class Constraint:
    kind: str
    before: str
    after: str
    condition: str = ""
    note: str = ""
    frames: tuple[str, ...] = ()


@dataclass(frozen=True)
class Binding:
    role: str
    object_id: str | None
    reason: str = ""  # why it did not bind; empty when bound


@dataclass(frozen=True)
class DemoExtraction:
    """The stage-1 description (kept for display) plus how the demo was produced and bound."""

    demo: str
    strategy: str  # "manual" keyframes (as in the paper) or "sampled" frames
    source: str  # "frames" | "video"
    frame_ids: tuple[str, ...]
    model: str
    scripted: bool  # True when no vision model looked at the frames (ReplayDemoModel)
    changes: tuple[ObjectChange, ...] = ()
    subtasks: tuple[Subtask, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    bindings: dict[str, Binding] = field(default_factory=dict)
    repairs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "demo": self.demo, "strategy": self.strategy, "source": self.source, "frames": list(self.frame_ids),
            "model": self.model, "scripted": self.scripted, "repairs": self.repairs,
            "evidence": "demonstration only: historical frames, not a hardware run and not evidence about the "
                        "current scene",
            "object_changes": [{"object": c.object, "change": c.change, "frames": list(c.frames)}
                               for c in self.changes],
            "subtasks": [{"id": s.id, "action": s.action, "object": s.object, "destination": s.destination,
                          "frames": list(s.frames), "confidence": s.confidence, "uncertain": s.uncertain,
                          "hidden": s.hidden, "transcript": s.transcript, "note": s.note} for s in self.subtasks],
            "constraints": [{"kind": c.kind, "before": c.before, "after": c.after, "condition": c.condition,
                             "note": c.note, "frames": list(c.frames)} for c in self.constraints],
            "bindings": {r: {"object_id": b.object_id, "reason": b.reason} for r, b in self.bindings.items()},
        }


# -- stage-1 parsing -----------------------------------------------------------------------


def _s(value: Any, where: str, *, required: bool = True, limit: int = MAX_TEXT) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()) or len(value) > limit:
        raise DemoError(f"{where}: expected a short string")
    if any(ord(c) < 32 for c in value):
        raise DemoError(f"{where}: control characters are not allowed")
    return value


def _obj(d: Any, allowed: set[str], where: str, required: Sequence[str] = ()) -> dict[str, Any]:
    if not isinstance(d, dict):
        raise DemoError(f"{where}: expected an object")
    unknown = set(d) - allowed
    if unknown:
        raise DemoError(f"{where}: unknown fields {sorted(unknown)}")
    missing = [k for k in required if k not in d]
    if missing:
        raise DemoError(f"{where}: missing fields {missing}")
    return d


def _frames(value: Any, where: str, demo: Demo, *, required: bool) -> tuple[str, ...]:
    if value is None:
        value = []
    if not isinstance(value, list) or len(value) > 16:
        raise DemoError(f"{where}: expected a short list of frame ids")
    if required and not value:
        raise DemoError(f"{where}: cite at least one demo frame")
    out: list[str] = []
    for ref in value:
        full = demo.resolve_frame(ref)
        if full not in out:
            out.append(full)
    return tuple(out)


def _flag(value: Any, where: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise DemoError(f"{where}: expected true or false")
    return value


def parse_description(data: Any, demo: Demo, *, model: str = "", scripted: bool = False) -> DemoExtraction:
    """Validate a stage-1 reply. Every cited frame must be one of the loaded demo frames."""
    d = _obj(data, {"object_changes", "subtasks", "constraints"}, "description", ("subtasks",))
    changes = []
    for i, raw in enumerate(d.get("object_changes") or []):
        c = _obj(raw, {"object", "change", "frames"}, f"object_changes[{i}]", ("object", "change"))
        changes.append(ObjectChange(_s(c["object"], f"object_changes[{i}].object"),
                                    _s(c["change"], f"object_changes[{i}].change"),
                                    _frames(c.get("frames"), f"object_changes[{i}].frames", demo, required=False)))
    raw_subtasks = d["subtasks"]
    if not isinstance(raw_subtasks, list) or len(raw_subtasks) > MAX_STEPS:
        raise DemoError(f"description.subtasks: expected a list of at most {MAX_STEPS}")
    subtasks: list[Subtask] = []
    for i, raw in enumerate(raw_subtasks):
        where = f"subtasks[{i}]"
        s = _obj(raw, {"id", "action", "object", "destination", "frames", "confidence", "uncertain", "hidden",
                       "transcript", "note"}, where, ("id", "action", "object"))
        sid = _s(s["id"], f"{where}.id", limit=40)
        if any(sid == t.id for t in subtasks):
            raise DemoError(f"{where}.id: duplicate subtask id {sid!r}")
        conf = s.get("confidence")
        if conf is not None and (isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1):
            raise DemoError(f"{where}.confidence: expected a number in [0, 1]")
        subtasks.append(Subtask(
            sid, _s(s["action"], f"{where}.action"), _s(s["object"], f"{where}.object"),
            _s(s.get("destination"), f"{where}.destination", required=False),
            _frames(s.get("frames"), f"{where}.frames", demo, required=True),
            None if conf is None else float(conf), _flag(s.get("uncertain"), f"{where}.uncertain"),
            _flag(s.get("hidden"), f"{where}.hidden"), _s(s.get("transcript"), f"{where}.transcript", required=False),
            _s(s.get("note"), f"{where}.note", required=False)))
    order = {s.id: i for i, s in enumerate(subtasks)}
    constraints = []
    for i, raw in enumerate(d.get("constraints") or []):
        where = f"constraints[{i}]"
        c = _obj(raw, {"kind", "before", "after", "condition", "note", "frames"}, where, ("kind", "before", "after"))
        kind = _s(c["kind"], f"{where}.kind", limit=20)
        if kind not in CONSTRAINT_KINDS:
            raise DemoError(f"{where}.kind: expected one of {CONSTRAINT_KINDS}")
        before, after = _s(c["before"], f"{where}.before", limit=40), _s(c["after"], f"{where}.after", limit=40)
        if before not in order or after not in order:
            raise DemoError(f"{where}: before/after must be subtask ids")
        if order[before] >= order[after]:
            raise DemoError(f"{where}: `before` must come earlier in the demonstrated order than `after`")
        condition = _s(c.get("condition"), f"{where}.condition", required=False, limit=80)
        if kind == "support" and not condition:
            raise DemoError(f"{where}: a support constraint needs a condition")
        constraints.append(Constraint(kind, before, after, condition, _s(c.get("note"), f"{where}.note",
                                                                          required=False),
                                      _frames(c.get("frames"), f"{where}.frames", demo, required=False)))
    return DemoExtraction(demo.name, demo.strategy, demo.source, demo.frame_ids(), model or "unknown", scripted,
                          tuple(changes), tuple(subtasks), tuple(constraints))


def describe_request(demo: Demo) -> str:
    shape = {
        "object_changes": [{"object": "the red block", "change": "table -> box", "frames": ["f000", "f002"]}],
        "subtasks": [{"id": "d1", "action": "put into", "object": "the red block", "destination": "the white box",
                      "frames": ["f001", "f002"], "confidence": 0.9, "uncertain": False, "hidden": False,
                      "transcript": "phrase from the transcript, or empty", "note": ""}],
        "constraints": [{"kind": "order", "before": "d1", "after": "d2", "condition": "", "note": "", "frames": []}]}
    return json.dumps({
        "demo": demo.name, "frame_strategy": demo.strategy,
        "frames": [{"image_number": i + 1, "id": f.frame_id, "t_s": round(f.t_s, 2)}
                   for i, f in enumerate(demo.frames)],
        "transcript": demo.transcript, "response_shape": shape}, ensure_ascii=False)


# -- role binding --------------------------------------------------------------------------

_STOP = {"the", "a", "an", "on", "of", "at", "to", "in", "side", "one"}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def bind_role(role: str, objects: dict[str, str] | Any) -> Binding:
    """Bind a demo role phrase to a CURRENT object id by label. Ambiguous or missing -> unbound, never guessed."""
    want = _tokens(role)
    if not want:
        return Binding(role, None, "empty role phrase")
    names = {oid: (_tokens(label), _tokens(oid.replace("_", " "))) for oid, label in objects.items()}
    exact = [oid for oid, (lab, ident) in names.items() if lab == want or ident == want]
    if len(exact) == 1:
        return Binding(role, exact[0])
    cands = exact or [oid for oid, (lab, ident) in names.items() if set(want) <= set(lab) | set(ident)]
    if len(cands) == 1:
        return Binding(role, cands[0])
    if not cands:
        return Binding(role, None, f"no object in the current scene is called {role!r}")
    labels = ", ".join(f"{objects[c]!r}" for c in sorted(cands))
    return Binding(role, None, f"{role!r} is ambiguous in the current scene (matches {labels})")


def bind_roles(extraction: DemoExtraction, world: SkillWorld) -> dict[str, Binding]:
    roles: list[str] = []
    for s in extraction.subtasks:
        for role in (s.object, s.destination):
            if role and role not in roles:
                roles.append(role)
    return {r: bind_role(r, world.objects) for r in roles}


# -- stage 2 -------------------------------------------------------------------------------


def map_request(demo: Demo, extraction: DemoExtraction, world: SkillWorld) -> str:
    skills = [{"skill": cap.skill_id, "arms": [a for a in cap.arms if a in world.targets()],
               "effect_template": cap.effect_templates[0] if cap.effect_templates else None,
               "description": cap.prompt_template} for cap in world.registry.skills.values()]
    shape = {"schema_version": SCHEMA_VERSION, "goal_text": f"repeat demo {demo.name}", "source_type": "demo_video",
             "steps": [{"id": "d1", "skill": "registered skill", "object_id": "bound id",
                        "destination_id": "bound id or empty", "arm": "allowed arm or null", "depends_on": [],
                        "effect": "the skill's effect_template filled in", "confidence": 0.9,
                        "sources": [{"kind": "demo_frame", "ref": "demo:<name>:f001"}]}],
             "obligations": ["effects that define done"], "unresolved": ["d3: ..."], "unsupported": ["d2: ..."]}
    description = extraction.to_dict()
    return json.dumps({
        "description": {k: description[k] for k in ("object_changes", "subtasks", "constraints")},
        "bindings": {r: b.object_id for r, b in extraction.bindings.items()},
        "objects": dict(world.objects), "skills": skills, "arms": sorted(world.targets()),
        "response_shape": shape}, ensure_ascii=False)


def _clip(text: str) -> str:
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT - 1] + "…"


def _string_list(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value) or len(value) > 2 * MAX_STEPS:
        raise DemoError(f"{where}: expected a bounded list of strings")
    return [_clip(v) for v in value]


def _finalize(data: Any, demo: Demo, ex: DemoExtraction, world: SkillWorld) -> ProcedureProposal:
    """Trusted post-processing of a stage-2 reply; raises ProcedureError for the repair loop."""
    if not isinstance(data, dict):
        raise DemoError("procedure: expected a JSON object")
    sub = {s.id: s for s in ex.subtasks}
    steps_raw = data.get("steps")
    if not isinstance(steps_raw, list) or len(steps_raw) > MAX_STEPS:
        raise DemoError("procedure.steps: expected a bounded list")
    raw_by_id: dict[str, dict[str, Any]] = {}
    for i, raw in enumerate(steps_raw):
        if not isinstance(raw, dict) or raw.get("id") not in sub:
            raise DemoError(f"procedure.steps[{i}].id: must be one of the stage-1 subtask ids {sorted(sub)}")
        unknown = set(raw) - _STEP_KEYS
        if unknown:
            raise DemoError(f"procedure.steps[{i}]: unknown fields {sorted(unknown)}")
        if raw["id"] in raw_by_id:
            raise DemoError(f"procedure.steps[{i}].id: duplicate {raw['id']!r}")
        raw_by_id[raw["id"]] = raw

    unresolved: list[tuple[str, str]] = []  # (subtask id, trusted report)
    unsupported: dict[str, str] = {}
    kept: list[str] = []
    for s in ex.subtasks:  # demonstrated order
        raw = raw_by_id.get(s.id)
        obj_b = ex.bindings[s.object]
        dest_b = ex.bindings.get(s.destination) if s.destination else None
        if s.hidden:
            unresolved.append((s.id, f"{s.id}: hidden action not mapped ({s.action} {s.object}): "
                                     f"{s.note or 'not visible'}"))
            continue
        unbound = next((b for b in (obj_b, dest_b) if b is not None and b.object_id is None), None)
        if unbound is not None:
            unresolved.append((s.id, f"{s.id}: cannot bind role: {unbound.reason}"))
            continue
        if raw is None:
            dest = f" -> {s.destination}" if s.destination else ""
            unsupported[s.id] = f"{s.id}: {s.action} {s.object}{dest} (no registered skill for it)"
            continue
        if raw.get("object_id") != obj_b.object_id:
            raise DemoError(f"step {s.id}: object_id must be {obj_b.object_id!r} (the binding of {s.object!r})")
        want_dest = dest_b.object_id if dest_b else ""
        if (raw.get("destination_id") or "") != (want_dest or ""):
            raise DemoError(f"step {s.id}: destination_id must be {want_dest!r} (the binding of {s.destination!r})")
        kept.append(s.id)

    # A support constraint is only safe to drop-through if its supporter survived; otherwise drop the step.
    changed = True
    while changed:
        changed = False
        for c in ex.constraints:
            if c.kind == "support" and c.after in kept and c.before not in kept:
                kept.remove(c.after)
                unresolved.append((c.after, f"{c.after}: needs {c.condition!r} held by {c.before}, which was not "
                                            "mapped to a skill; step not mapped"))
                changed = True

    pos = {s.id: i for i, s in enumerate(ex.subtasks)}
    steps_out: list[dict[str, Any]] = []
    for sid in kept:
        s, raw = sub[sid], raw_by_id[sid]
        depends = [d for d in (raw.get("depends_on") or []) if d in kept and pos.get(d, 99) < pos[sid]]
        maintain: list[str] = []
        for c in ex.constraints:
            if c.after != sid:
                continue
            if c.before in kept and c.before not in depends:
                depends.append(c.before)
            elif c.before not in kept and c.kind == "order":
                unresolved.append((sid, f"{sid}: follows {c.before} in the demo, which was not mapped to a skill"))
            if c.kind == "support" and c.condition not in maintain:
                maintain.append(c.condition)
        sources = [{"kind": "demo_frame", "ref": f, "note": ""} for f in s.frames]
        for j, src in enumerate(raw.get("sources") or []):
            if not isinstance(src, dict) or src.get("kind") not in ("demo_frame", "demo_transcript"):
                raise DemoError(f"step {sid}.sources[{j}]: kind must be demo_frame or demo_transcript")
            if src["kind"] == "demo_frame":
                ref = demo.resolve_frame(src.get("ref"))
                if all(x["ref"] != ref for x in sources):
                    sources.append({"kind": "demo_frame", "ref": ref, "note": ""})
        if s.transcript:
            sources.append({"kind": "demo_transcript", "ref": f"demo:{demo.name}:transcript",
                            "note": _clip(s.transcript)})
        conf = raw.get("confidence")
        if conf is None:
            conf = s.confidence if s.confidence is not None else DEFAULT_CONFIDENCE
        if s.uncertain and isinstance(conf, (int, float)) and not isinstance(conf, bool):
            conf = min(conf, UNCERTAIN_CONFIDENCE_CAP)
            unresolved.append((sid, f"{sid}: demonstrator action is uncertain ({s.action} {s.object}); "
                                    f"confirm before running. {s.note}".strip()))
        step = {k: v for k, v in raw.items() if k in {"id", "skill", "object_id", "destination_id", "arm", "effect"}}
        step.update(depends_on=depends, maintain=maintain, sources=sources, confidence=conf, note=_clip(s.note))
        steps_out.append(step)

    effects = []
    for step in steps_out:  # obligations = every demonstrated effect that survived mapping
        skill = world.registry.skills.get(step["skill"]) if isinstance(step.get("skill"), str) else None
        try:
            effects.append(world.effect_for(skill, step["object_id"], step.get("destination_id") or "")
                           if skill else "invalid")
        except (ProcedureError, KeyError):  # parse_procedure reports the real problem below
            effects.append("invalid")

    covered = {sid for sid, _ in unresolved} | set(unsupported)
    extra_unresolved = [m for m in _string_list(data.get("unresolved"), "procedure.unresolved")
                        if m.split(":", 1)[0] not in covered]
    extra_unsupported = [m for m in _string_list(data.get("unsupported"), "procedure.unsupported")
                         if m.split(":", 1)[0] not in covered]
    proposal = parse_procedure({
        "schema_version": SCHEMA_VERSION, "goal_text": f"repeat the demonstrated procedure {demo.name}",
        "source_type": "demo_video", "steps": steps_out, "obligations": effects,
        "unresolved": ([m for _, m in unresolved] + extra_unresolved)[:2 * MAX_STEPS],
        "unsupported": (list(unsupported.values()) + extra_unsupported)[:2 * MAX_STEPS]}, world)
    return replace(proposal, source_type="demo_video")


def describe_demo(demo: Demo, model: TextModel, *, max_repairs: int = 1) -> DemoExtraction:
    """Stage 1 alone: demo frames + transcript -> validated description."""
    user = describe_request(demo)
    label = getattr(model, "label", type(model).__name__)
    scripted = bool(getattr(model, "scripted", False))
    images = demo.model_images()
    error = ""
    for attempt in range(max_repairs + 1):
        reply = model.complete(DESCRIBE_SYSTEM, user + (f"\nYOUR PREVIOUS REPLY WAS INVALID: {error}" if error else ""),
                               images)
        try:
            ex = parse_description(extract_json(reply), demo, model=label, scripted=scripted)
            return replace(ex, repairs=attempt)
        except _REPLY_ERRORS as e:
            error = str(e)[:500]
    raise DemoError(f"demo description failed: {error}")


def ground_demo(demo: Demo, world: SkillWorld, model: TextModel, *,
                max_repairs: int = 1) -> tuple[ProcedureProposal, DemoExtraction]:
    """Demo -> validated ``ProcedureProposal`` (source_type ``demo_video``) plus the stage-1 description."""
    extraction = describe_demo(demo, model, max_repairs=max_repairs)
    extraction = replace(extraction, bindings=bind_roles(extraction, world))
    user = map_request(demo, extraction, world)
    error = ""
    for attempt in range(max_repairs + 1):
        reply = model.complete(MAP_SYSTEM, user + (f"\nYOUR PREVIOUS REPLY WAS INVALID: {error}" if error else ""))
        try:
            proposal = _finalize(extract_json(reply), demo, extraction, world)
            return proposal, replace(extraction, repairs=extraction.repairs + attempt)
        except _REPLY_ERRORS as e:
            error = str(e)[:500]
    raise DemoError(f"demo mapping failed: {error}")
