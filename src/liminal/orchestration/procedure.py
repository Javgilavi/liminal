"""One symbolic procedure representation for every input adapter (WS3, V2/V4).

A free-text goal, a demonstration video, or a retrieved past procedure all become a
``ProcedureProposal``: ordered atomic steps over registered skills and known objects, plus
the obligations that define "done". It is a *proposal*. Trusted code here validates it
against the runtime capability registry (closed sets only), renders the FLUX instruction for
each step from the skill's template, and either

* turns its obligations into the trusted ``GoalSpec`` for the planner loop (``run_episode``), or
* compiles it into a reactive authored tree (``run_authored``): every step is
  ``retry{fallback[check effect, skill]}``, so already-true effects are skipped and failed or
  falsely-reported steps are retried with a fresh attempt.

A model can never add a skill, an object, executable code or a physical fact through this
path; anything it cannot map is reported in ``unsupported`` / ``unresolved`` instead.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from liminal.orchestration.plan_validation import CapabilityRegistry, GoalSpec, SkillCapability
from liminal.orchestration.task_spec import TaskSpec, parse_task

SCHEMA_VERSION = 1
SOURCE_TYPES = ("text", "demo_video", "experience")
MAX_STEPS = 16
MAX_TEXT = 240

_TOP_FIELDS = {"schema_version", "goal_text", "source_type", "steps", "obligations", "unresolved", "unsupported"}
_STEP_FIELDS = {"id", "skill", "object_id", "destination_id", "arm", "depends_on", "effect", "maintain",
                "sources", "confidence", "note"}
_SOURCE_FIELDS = {"kind", "ref", "note"}


class ProcedureError(ValueError):
    """A procedure proposal failed a schema, capability or authority check."""


@dataclass(frozen=True)
class SourceRef:
    """Where a step came from: a goal phrase, a demo frame/time, or a past run."""

    kind: str
    ref: str
    note: str = ""


@dataclass(frozen=True)
class ProcedureStep:
    step_id: str
    skill: str
    object_id: str
    destination_id: str
    effect: str
    arm: str | None = None  # a suggestion; trusted code picks an allowed arm when absent
    depends_on: tuple[str, ...] = ()
    maintain: tuple[str, ...] = ()  # conditions that must stay true while this step runs (V5)
    sources: tuple[SourceRef, ...] = ()
    confidence: float | None = None
    note: str = ""


@dataclass(frozen=True)
class ProcedureProposal:
    goal_text: str
    source_type: str
    steps: tuple[ProcedureStep, ...]
    obligations: tuple[str, ...]
    unresolved: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "goal_text": self.goal_text, "source_type": self.source_type,
            "obligations": list(self.obligations), "unresolved": list(self.unresolved),
            "unsupported": list(self.unsupported),
            "steps": [{
                "id": s.step_id, "skill": s.skill, "object_id": s.object_id, "destination_id": s.destination_id,
                "arm": s.arm, "depends_on": list(s.depends_on), "effect": s.effect, "maintain": list(s.maintain),
                "sources": [{"kind": r.kind, "ref": r.ref, "note": r.note} for r in s.sources],
                "confidence": s.confidence, "note": s.note,
            } for s in self.steps],
        }


@dataclass(frozen=True)
class SkillWorld:
    """The closed sets a procedure may refer to: known objects and the runtime registry."""

    objects: Mapping[str, str]
    registry: CapabilityRegistry

    def targets(self) -> set[str]:
        targets = set(self.registry.available_arms)
        targets.update(g for g, members in self.registry.arm_groups.items()
                       if members and all(m in self.registry.available_arms for m in members))
        return targets

    def effect_for(self, skill: SkillCapability, object_id: str, destination_id: str) -> str:
        if not skill.effect_templates:
            raise ProcedureError(f"skill {skill.skill_id!r} has no verifiable effect")
        return skill.effect_templates[0].format(object_id=object_id, destination_id=destination_id)

    def arm_for(self, skill: SkillCapability, requested: str | None) -> str:
        allowed = [a for a in skill.arms if a in self.targets()]
        if not allowed:
            raise ProcedureError(f"skill {skill.skill_id!r} has no available arm")
        if requested is None:
            return allowed[0]
        if requested not in allowed:
            raise ProcedureError(f"arm {requested!r} cannot run {skill.skill_id!r} (allowed: {allowed})")
        return requested

    def needs_destination(self, skill: SkillCapability) -> bool:
        return any("{destination_id}" in t for t in skill.effect_templates)

    def needs_object(self, skill: SkillCapability) -> bool:
        """State skills (e.g. open_lid -> lid_open) have no object fields in their effect."""
        return any("{object_id}" in t for t in skill.effect_templates)

    def render_prompt(self, skill: SkillCapability, object_id: str, destination_id: str) -> str:
        return skill.prompt_template.format(
            skill_id=skill.skill_id, object_id=object_id, destination_id=destination_id,
            object_label=self.objects.get(object_id, ""), destination_label=self.objects.get(destination_id, ""))


# -- validation --------------------------------------------------------------------------


def _text(value: Any, where: str, *, required: bool = True, limit: int = MAX_TEXT) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()) or len(value) > limit:
        raise ProcedureError(f"{where}: expected a short string")
    if any(ord(c) < 32 for c in value):
        raise ProcedureError(f"{where}: control characters are not allowed")
    return value


def _texts(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > MAX_STEPS * 2:
        raise ProcedureError(f"{where}: expected a bounded list")
    return tuple(_text(v, f"{where}[{i}]") for i, v in enumerate(value))


def _exact(d: Any, allowed: set[str], where: str, required: Iterable[str] = ()) -> dict[str, Any]:
    if not isinstance(d, dict):
        raise ProcedureError(f"{where}: expected an object")
    unknown = set(d) - allowed
    if unknown:
        raise ProcedureError(f"{where}: unknown fields {sorted(unknown)}")
    missing = [k for k in required if k not in d]
    if missing:
        raise ProcedureError(f"{where}: missing fields {missing}")
    return d


def parse_procedure(data: Any, world: SkillWorld) -> ProcedureProposal:
    """Validate an untrusted proposal (e.g. model JSON) against the closed skill/object sets."""
    d = _exact(data, _TOP_FIELDS, "procedure", ("steps", "obligations"))
    if d.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ProcedureError("procedure: unsupported schema_version")
    source_type = d.get("source_type", "text")
    if source_type not in SOURCE_TYPES:
        raise ProcedureError(f"procedure.source_type must be one of {SOURCE_TYPES}")
    raw_steps = d["steps"]
    if not isinstance(raw_steps, list) or len(raw_steps) > MAX_STEPS:
        raise ProcedureError(f"procedure.steps: expected a list of at most {MAX_STEPS} steps")
    steps: list[ProcedureStep] = []
    seen: set[str] = set()
    effects: set[str] = set()
    for i, raw in enumerate(raw_steps):
        where = f"procedure.steps[{i}]"
        s = _exact(raw, _STEP_FIELDS, where, ("id", "skill"))
        step_id = _text(s["id"], f"{where}.id", limit=40)
        if step_id in seen:
            raise ProcedureError(f"{where}.id: duplicate step id {step_id!r}")
        skill = world.registry.skills.get(s["skill"]) if isinstance(s["skill"], str) else None
        if skill is None:
            raise ProcedureError(f"{where}.skill: unregistered skill {s['skill']!r}")
        needs_object = world.needs_object(skill)
        object_id = _text(s.get("object_id", ""), f"{where}.object_id", required=needs_object, limit=40)
        destination_id = _text(s.get("destination_id", ""), f"{where}.destination_id", required=False, limit=40)
        if (needs_object or object_id) and object_id not in world.objects:
            raise ProcedureError(f"{where}.object_id: unknown object {object_id!r}")
        if destination_id and destination_id not in world.objects:
            raise ProcedureError(f"{where}.destination_id: unknown object {destination_id!r}")
        if destination_id and destination_id == object_id:
            raise ProcedureError(f"{where}: object and destination must differ")
        if world.needs_destination(skill) and not destination_id:
            raise ProcedureError(f"{where}.destination_id: {skill.skill_id!r} needs a known destination")
        effect = world.effect_for(skill, object_id, destination_id)
        if "effect" in s and s["effect"] != effect:
            raise ProcedureError(f"{where}.effect: expected the registered effect {effect!r}, got {s['effect']!r}")
        if effect in effects:
            raise ProcedureError(f"{where}.effect: {effect!r} is produced by more than one step")
        arm = s.get("arm")
        if arm is not None and not isinstance(arm, str):
            raise ProcedureError(f"{where}.arm: expected a string or null")
        arm = world.arm_for(skill, arm or None)
        depends = _texts(s.get("depends_on"), f"{where}.depends_on")
        missing = [x for x in depends if x not in seen]
        if missing:
            raise ProcedureError(f"{where}.depends_on: unknown or later steps {missing} (list steps in order)")
        maintain = _texts(s.get("maintain"), f"{where}.maintain")
        sources = []
        for j, src in enumerate(s.get("sources") or []):
            r = _exact(src, _SOURCE_FIELDS, f"{where}.sources[{j}]", ("kind", "ref"))
            sources.append(SourceRef(_text(r["kind"], f"{where}.sources[{j}].kind", limit=20),
                                     _text(r["ref"], f"{where}.sources[{j}].ref", limit=80),
                                     _text(r.get("note"), f"{where}.sources[{j}].note", required=False)))
        confidence = s.get("confidence")
        if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                                       or not 0 <= confidence <= 1):
            raise ProcedureError(f"{where}.confidence: expected a number in [0, 1]")
        steps.append(ProcedureStep(step_id, skill.skill_id, object_id, destination_id, effect, arm, depends,
                                   maintain, tuple(sources), confidence,
                                   _text(s.get("note"), f"{where}.note", required=False)))
        seen.add(step_id)
        effects.add(effect)
    obligations = _texts(d["obligations"], "procedure.obligations")
    if steps and not obligations:
        raise ProcedureError("procedure.obligations: at least one obligation is required")
    stray = [o for o in obligations if o not in effects]
    if stray:
        raise ProcedureError(f"procedure.obligations: {stray} are not produced by any step")
    if len(set(obligations)) != len(obligations):
        raise ProcedureError("procedure.obligations: duplicates")
    return ProcedureProposal(
        goal_text=_text(d.get("goal_text", "") or "(unspecified)", "procedure.goal_text", limit=1000),
        source_type=source_type, steps=tuple(steps), obligations=obligations,
        unresolved=_texts(d.get("unresolved"), "procedure.unresolved"),
        unsupported=_texts(d.get("unsupported"), "procedure.unsupported"))


# -- trusted outputs ---------------------------------------------------------------------


def describe(proposal: ProcedureProposal, world: SkillWorld) -> list[dict[str, Any]]:
    """The atomic-skill breakdown an operator reviews: one row per FLUX call."""
    rows = []
    for step in proposal.steps:
        skill = world.registry.skills[step.skill]
        rows.append({
            "id": step.step_id, "skill": step.skill, "arm": step.arm,
            "object": world.objects.get(step.object_id, ""), "object_id": step.object_id,
            "destination": world.objects.get(step.destination_id, ""), "destination_id": step.destination_id,
            "effect": step.effect, "obligation": step.effect in proposal.obligations,
            "flux_prompt": world.render_prompt(skill, step.object_id, step.destination_id),
            "depends_on": list(step.depends_on), "maintain": list(step.maintain),
            "sources": [{"kind": r.kind, "ref": r.ref, "note": r.note} for r in step.sources],
            "confidence": step.confidence, "note": step.note,
        })
    return rows


def goal_spec(proposal: ProcedureProposal, goal_id: str) -> GoalSpec:
    """Only call after an operator (or a trusted fixture) accepted the obligations."""
    if not proposal.obligations:
        raise ProcedureError("procedure has no obligations to turn into a goal")
    return GoalSpec(goal_id, proposal.goal_text, proposal.obligations)


def compile_procedure(proposal: ProcedureProposal, world: SkillWorld, name: str = "procedure") -> TaskSpec:
    """Reactive authored tree: skip already-true effects, retry with a fresh attempt, keep order.

    A step with a maintained condition C becomes a cooperative block (V5): the earlier step whose
    effect is C establishes it on the support arm, a ``while`` holds that arm and re-observes C while
    this step runs on the other arm, and an explicit ``release`` follows. Losing C fails the block and
    the retry re-establishes it (its guard skips work that is still true).
    """
    if not proposal.steps:
        raise ProcedureError("procedure has no steps")
    by_effect = {step.effect: step for step in proposal.steps}
    supports: dict[str, ProcedureStep] = {}  # establishing step id -> the step that needs it held
    for step in proposal.steps:
        if len(step.maintain) > 1:
            raise ProcedureError(f"step {step.step_id}: one maintained condition per step is supported")
        for condition in step.maintain:
            est = by_effect.get(condition)
            if est is None or proposal.steps.index(est) > proposal.steps.index(step):
                raise ProcedureError(f"step {step.step_id}: maintained {condition!r} needs an earlier step "
                                     "that establishes it")
            if world.targets() and set(_members(world, est.arm)) & set(_members(world, step.arm)):
                raise ProcedureError(f"step {step.step_id}: the support arm cannot also run the step")
            if est.step_id in supports:
                raise ProcedureError(f"step {est.step_id} already supports another step")
            supports[est.step_id] = step

    def leaf(step: ProcedureStep) -> dict[str, Any]:
        skill = world.registry.skills[step.skill]
        params: dict[str, Any] = {"resources": list(skill.resources)}
        if step.object_id:
            params.update(object_id=step.object_id, destination_id=step.destination_id)
        body: dict[str, Any] = {"arm": step.arm, "name": step.skill, "prompt": world.render_prompt(
            skill, step.object_id, step.destination_id), "success": step.effect, "timeout_s": skill.max_timeout_s,
            "params": params}
        if step.object_id:
            body["target"] = step.object_id
        return {"skill": body}

    def guarded(step: ProcedureStep) -> dict[str, Any]:
        skill = world.registry.skills[step.skill]
        node: dict[str, Any] = {"fallback": [{"check": step.effect}, leaf(step)], "name": f"step:{step.step_id}"}
        if skill.preconditions:
            node = {"sequence": [*({"check": p} for p in skill.preconditions), node]}
        return node

    def retried(step: ProcedureStep, node: dict[str, Any], **extra: Any) -> dict[str, Any]:
        attempts = max(1, world.registry.skills[step.skill].max_attempts, extra.pop("min_attempts", 1))
        return {"retry": {"attempts": attempts, "do": node, **extra}} if attempts > 1 else node

    nodes: list[dict[str, Any]] = []
    for step in proposal.steps:
        if step.step_id in supports:
            continue  # compiled inside the cooperative block of the step it supports
        if not step.maintain:
            nodes.append(retried(step, guarded(step)))
            continue
        est = by_effect[step.maintain[0]]
        block = {"sequence": [guarded(est), {"while": {
            "arm": est.arm, "maintain": step.maintain[0], "until": step.effect, "do": guarded(step)}}],
            "name": f"hold:{est.step_id}+{step.step_id}"}
        home = {"parallel": [{"skill": {"arm": a, "name": "home"}} for a in (est.arm, step.arm)]}
        nodes.append(retried(step, block, min_attempts=3, recovery=home))
        nodes.append({"skill": {"arm": est.arm, "name": "release"}})
    data = {"name": name, "description": proposal.goal_text, "goal": proposal.obligations[-1],
            "tree": nodes[0] if len(nodes) == 1 else {"sequence": nodes}}
    from liminal.orchestration.skills import PrimitiveBackend

    # Primitives (home/release/hold) run on the local controller, never through FLUX.
    return parse_task(data, arms=world.targets(), skills={*world.registry.skills, *PrimitiveBackend.NAMES})


def _members(world: SkillWorld, arm: str | None) -> tuple[str, ...]:
    if arm is None:
        return ()
    return tuple(world.registry.arm_groups.get(arm, (arm,)))


def scenario_with_procedure(scenario: Any, proposal: ProcedureProposal, goal_id: str = "goal-custom") -> Any:
    """Re-target a replay scenario at an accepted procedure (same scene, new trusted goal)."""
    obligations = {s.effect: {"object_id": s.object_id, "destination_id": s.destination_id}
                   for s in proposal.steps if s.effect in proposal.obligations and s.object_id}
    return replace(scenario, goal=goal_spec(proposal, goal_id), obligations=obligations)


# -- text goal grounding -----------------------------------------------------------------


class TextModel(Protocol):
    def complete(self, system: str, user: str, images: Sequence[bytes] = ()) -> str: ...


GROUNDING_SYSTEM = """You turn an operator's goal for a dual-arm robot into a symbolic procedure.
Use ONLY the listed skills, object ids and arms. Each step is one atomic, verifiable skill call
on one object (e.g. one pick_place). Order steps as they should run; list dependencies by step id.
Put every effect that defines "done" in "obligations". If a step needs another step's effect to stay
true while it runs (e.g. one arm holds a lid open while the other inserts), list that effect in the
step's "maintain" and run the two steps on different arms. If part of the goal cannot be expressed with
the listed skills, do not invent a skill: describe it in "unsupported". Put ambiguities in
"unresolved". Reply with ONE JSON object and nothing else, matching "response_shape"."""


def grounding_request(text: str, world: SkillWorld, source_type: str = "text") -> str:
    skills = [{"skill": cap.skill_id, "arms": [a for a in cap.arms if a in world.targets()],
               "effect_template": cap.effect_templates[0] if cap.effect_templates else None,
               "description": cap.prompt_template} for cap in world.registry.skills.values()]
    shape = {"schema_version": SCHEMA_VERSION, "goal_text": "the goal", "source_type": source_type,
             "steps": [{"id": "s1", "skill": "registered skill", "object_id": "known id",
                        "destination_id": "known id or empty", "arm": "allowed arm or null",
                        "depends_on": [], "effect": "the skill's effect_template filled in", "maintain": [],
                        "sources": [{"kind": "text", "ref": "goal phrase"}]}],
             "obligations": ["effects that define done"], "unresolved": [], "unsupported": []}
    return json.dumps({"goal": text, "objects": dict(world.objects), "skills": skills,
                       "arms": sorted(world.targets()), "response_shape": shape}, ensure_ascii=False)


def extract_json(text: str) -> Any:
    """First top-level JSON object in a model reply (models sometimes add a code fence)."""
    start = text.find("{")
    if start < 0:
        raise ProcedureError("model reply contains no JSON object")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise ProcedureError(f"model reply is not valid JSON: {e}") from e
    return obj


def ground_goal(text: str, world: SkillWorld, model: TextModel, *, images: Sequence[bytes] = (),
                max_repairs: int = 1) -> ProcedureProposal:
    """Free-text goal -> validated procedure. Validation errors are fed back at most ``max_repairs`` times."""
    text = _text(text, "goal", limit=1000)
    user = grounding_request(text, world)
    error = ""
    for _ in range(max_repairs + 1):
        reply = model.complete(GROUNDING_SYSTEM, user + (f"\nYOUR PREVIOUS REPLY WAS INVALID: {error}" if error
                                                          else ""), images)
        try:
            proposal = parse_procedure(extract_json(reply), world)
            return replace(proposal, goal_text=text, source_type="text")
        except ProcedureError as e:
            error = str(e)[:500]
    raise ProcedureError(f"goal grounding failed: {error}")


_CONTAINER_WORDS = ("box", "bin", "bowl", "tray", "basket", "container", "plate", "cup", "drawer")
_UNSUPPORTED_VERBS = ("stack", "pour", "hand over", "handover", "rotate", "open", "close", "wipe", "screw", "push")


@dataclass
class KeywordGroundingModel:
    """Offline, deterministic grounder: no key, no network. Good enough for demos and tests.

    Splits the goal into clauses ("and", "then", ",", ";"), finds known object labels in each,
    treats container-like objects as destinations, and honours "left/right arm" hints.
    """

    world: SkillWorld
    calls: list[str] = field(default_factory=list)

    def complete(self, system: str, user: str, images: Sequence[bytes] = ()) -> str:
        self.calls.append(user)
        goal = json.loads(user.split("\nYOUR PREVIOUS REPLY", 1)[0])["goal"]
        return json.dumps(self.ground(goal))

    def _mentions(self, clause: str) -> list[tuple[int, str]]:
        found = []
        for object_id, label in self.world.objects.items():
            follow = r"(?=\s*(?:$|[,.;]|(?:ones?|cubes?|and|then|in|into|inside|on|to|with)\b))"
            patterns = [rf"\b{re.escape(label.lower())}s?\b"]
            # A bare id/colour ("red") means the red block only when no other noun follows ("red pencil" is not it).
            for short in {object_id.lower().replace("_", " "), label.lower().removesuffix(" block")} - {label.lower()}:
                patterns.append(rf"\b{re.escape(short)}\b{follow}")
            for pattern in patterns:
                m = re.search(pattern, clause)
                if m:
                    found.append((m.start(), object_id))
                    break
        return sorted(set(found))

    def _is_container(self, object_id: str) -> bool:
        text = f"{object_id} {self.world.objects[object_id]}".lower()
        return any(w in text for w in _CONTAINER_WORDS)

    def ground(self, goal: str) -> dict[str, Any]:
        skill = next(iter(self.world.registry.skills.values()))
        clauses = [c.strip() for c in re.split(r",|;|\bthen\b|\band\b", goal.lower()) if c.strip()]
        steps, unsupported, unresolved = [], [], []
        destination = ""
        pending_objects: list[str] = []
        for clause in clauses:
            if any(v in clause for v in _UNSUPPORTED_VERBS):
                unsupported.append(clause)
                continue
            mentions = [oid for _, oid in self._mentions(clause)]
            if not mentions:
                unresolved.append(f"not understood: no object of this scene in {clause!r} "
                                  f"(scene has: {', '.join(self.world.objects.values())})")
                continue
            dests = [oid for oid in mentions if self._is_container(oid)]
            objs = [oid for oid in mentions if not self._is_container(oid)]
            if not objs and any(w in clause for w in ("rest", "others", "remaining", "everything", "all")):
                unresolved.append(f"not understood: {clause!r} refers to objects by context; name them explicitly")
                continue
            if dests:
                destination = dests[-1]
            arm = "left" if "left arm" in clause else "right" if "right arm" in clause else None
            if not dests and objs and not destination:
                pending_objects.extend(objs)  # "the red block and the blue block in the box"
                continue
            for oid in pending_objects + objs:
                steps.append((oid, destination, arm, clause))
            pending_objects = []
        for oid in pending_objects:
            if destination and destination in self.world.objects:
                steps.append((oid, destination, None, goal))
            else:
                unresolved.append(f"no destination for {self.world.objects[oid]}")
        out_steps, effects, previous = [], [], None
        for i, (oid, dest, arm, clause) in enumerate(steps, 1):
            effect = self.world.effect_for(skill, oid, dest)
            if effect in effects:
                continue
            allowed = [a for a in skill.arms if a in self.world.targets()]
            out_steps.append({"id": f"s{i}", "skill": skill.skill_id, "object_id": oid, "destination_id": dest,
                              "arm": arm if arm in allowed else None, "effect": effect,
                              "depends_on": [previous] if previous and "then" in goal.lower() else [],
                              "sources": [{"kind": "text", "ref": clause[:80]}]})
            effects.append(effect)
            previous = f"s{i}"
        if not out_steps and not unsupported and not unresolved:
            unresolved.append("no known objects found in the goal")
        return {"schema_version": SCHEMA_VERSION, "goal_text": goal, "source_type": "text", "steps": out_steps,
                "obligations": effects, "unresolved": unresolved, "unsupported": unsupported}
