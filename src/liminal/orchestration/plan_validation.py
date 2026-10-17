"""Typed, bounded proposal contracts for model-assisted task planning.

The model output is an untrusted proposal.  This module binds it to a trusted
goal, observation revision and runtime capability registry before rendering the
existing task grammar.  It deliberately has no world-state write path.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from string import Formatter
from typing import Any

from liminal.orchestration.task_spec import TaskSpec, TaskSpecError, parse_task


class PlanValidationError(ValueError):
    """A proposal failed a schema, authority, capability or budget check."""


@dataclass(frozen=True)
class GoalSpec:
    """Trusted user goal and its independently established obligations."""

    goal_id: str
    text: str
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.goal_id, str) or not self.goal_id or not isinstance(self.text, str)
                or not self.text or not self.obligations
                or any(not isinstance(item, str) or not item for item in self.obligations)
                or len(set(self.obligations)) != len(self.obligations)):
            raise ValueError("GoalSpec requires an id, text and at least one obligation")


@dataclass(frozen=True)
class ObservationFrame:
    """Labeled planner image; historical frames cannot stand in for live state."""

    frame_id: str
    camera_role: str
    captured_at: float
    temporal_role: str
    media_type: str
    data: bytes = field(repr=False)
    revision: int | None = None

    def __post_init__(self) -> None:
        if self.temporal_role not in {"current", "historical"}:
            raise ValueError("temporal_role must be 'current' or 'historical'")
        if not self.frame_id or not self.camera_role or not self.media_type:
            raise ValueError("frame id, camera role and media type are required")
        if self.media_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
            raise ValueError("planner frames must use a supported image media type")
        if not isinstance(self.data, bytes):
            raise TypeError("frame data must be bytes")
        if (isinstance(self.captured_at, bool) or not isinstance(self.captured_at, (int, float))
                or not math.isfinite(self.captured_at)):
            raise ValueError("frame capture time must be finite")
        if self.revision is not None and (isinstance(self.revision, bool) or not isinstance(self.revision, int)):
            raise ValueError("frame revision must be an integer")


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    kind: str
    object_ids: tuple[str, ...] = ()
    predicate: str | None = None
    temporal_role: str = "current"
    revision: int | None = None


@dataclass(frozen=True)
class SkillCapability:
    """A runtime-registered skill and the effects it can verify.

    Effects are exact predicate templates, for example
    ``"{object_id}_in_{destination_id}"``.  A skill with no effect template
    cannot be used for scene-changing planning.
    """

    skill_id: str
    arms: tuple[str, ...]
    effect_templates: tuple[str, ...]
    prompt_template: str = "{skill_id} {object_label} {destination_label}"
    max_timeout_s: float = 30.0
    max_attempts: int = 1
    resources: tuple[str, ...] = ()
    changes_scene: bool = True
    preconditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.skill_id or not self.arms:
            raise ValueError("skill capability requires an id and at least one arm/group")
        if (isinstance(self.max_timeout_s, bool) or not isinstance(self.max_timeout_s, (int, float))
                or not math.isfinite(self.max_timeout_s) or self.max_timeout_s <= 0):
            raise ValueError("skill capability timeout must be finite and positive")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("skill capability attempt limit must be a positive integer")
        allowed_template_fields = {"skill_id", "object_id", "destination_id", "object_label", "destination_label"}
        fields = {name for _, name, _, _ in Formatter().parse(self.prompt_template) if name}
        if not fields.issubset(allowed_template_fields):
            raise ValueError(f"skill prompt template has unsupported fields {sorted(fields - allowed_template_fields)}")
        if not self.prompt_template or len(self.prompt_template) > 240:
            raise ValueError("skill prompt template must be non-empty and bounded")
        allowed_effect_fields = {"object_id", "destination_id"}
        effect_fields = {
            name for template in self.effect_templates
            for _, name, _, _ in Formatter().parse(template) if name
        }
        if not effect_fields.issubset(allowed_effect_fields):
            raise ValueError(f"effect template has unsupported fields {sorted(effect_fields - allowed_effect_fields)}")


@dataclass(frozen=True)
class CapabilityRegistry:
    skills: Mapping[str, SkillCapability]
    available_arms: tuple[str, ...]
    arm_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    version: str = "1"

    def __post_init__(self) -> None:
        for group, members in self.arm_groups.items():
            if (not isinstance(group, str) or not group or not members
                    or any(not isinstance(member, str) or not member for member in members)
                    or len(set(members)) != len(members)):
                raise ValueError(f"arm group {group!r} must have unique members")
        for skill_id, capability in self.skills.items():
            if skill_id != capability.skill_id:
                raise ValueError(f"capability key {skill_id!r} does not match its skill id")


@dataclass(frozen=True)
class PlanBudget:
    max_steps: int = 1
    max_seconds: float = 60.0
    max_attempts: int = 4
    max_model_calls: int = 3
    max_proposal_bytes: int = 16_384
    max_prompt_chars: int = 240
    max_context_frames: int = 8
    max_observation_bytes: int = 8 * 1024 * 1024
    max_sketch_items: int = 16

    def __post_init__(self) -> None:
        integers = (self.max_steps, self.max_attempts, self.max_model_calls,
                    self.max_proposal_bytes, self.max_prompt_chars, self.max_context_frames,
                    self.max_observation_bytes, self.max_sketch_items)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in integers):
            raise ValueError("plan count and size budgets must be positive integers")
        if (isinstance(self.max_seconds, bool) or not isinstance(self.max_seconds, (int, float))
                or not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("plan time budget must be finite and positive")


@dataclass(frozen=True)
class PlanningContext:
    run_id: str
    revision: int
    goal: GoalSpec
    observations: tuple[ObservationFrame, ...]
    evidence: tuple[EvidenceRecord, ...]
    object_labels: Mapping[str, str]
    capabilities: CapabilityRegistry
    budget: PlanBudget = field(default_factory=PlanBudget)
    progress: Mapping[str, Any] = field(default_factory=dict)
    # Advisory context only (never evidence, never progress): the verified-ledger record and keyframe notes
    # in ``memory``, retrieved cross-run items in ``experience``. ``nominate_keyframes`` asks the model for
    # MemER-style ``keyframe_nominations`` over the frames it is shown.
    memory: Mapping[str, Any] = field(default_factory=dict)
    experience: Mapping[str, Any] = field(default_factory=dict)
    nominate_keyframes: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("planning context requires a run id")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise ValueError("planning context revision must be a non-negative integer")
        if len(self.observations) > self.budget.max_context_frames:
            raise ValueError("observation frame count exceeds context budget")
        if sum(len(frame.data) for frame in self.observations) > self.budget.max_observation_bytes:
            raise ValueError("observation bytes exceed context budget")
        frame_ids = [frame.frame_id for frame in self.observations]
        evidence_ids = [record.evidence_id for record in self.evidence]
        if len(set(frame_ids)) != len(frame_ids) or len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("observation and evidence ids must be unique")
        for object_id, label in self.object_labels.items():
            if (not isinstance(object_id, str) or not object_id or not isinstance(label, str) or not label.strip()
                    or len(label) > 80 or any(ord(character) < 32 for character in label)):
                raise ValueError("object ids and labels must be short, printable strings")
        completed = self.progress.get("completed_effects", ())
        if not isinstance(completed, (list, tuple)) or any(not isinstance(item, str) for item in completed):
            raise ValueError("progress.completed_effects must be a list of predicate strings")


@dataclass(frozen=True)
class ValidatedPlan:
    run_id: str
    base_revision: int
    goal_id: str
    dependency_sketch: tuple[str, ...]
    task: TaskSpec
    step_ids: tuple[str, ...]
    used_evidence: tuple[str, ...]
    # Advisory keyframe nominations that named a frame the model was shown, and those that did not.
    keyframe_nominations: tuple[Mapping[str, str], ...] = ()
    rejected_nominations: tuple[Mapping[str, str], ...] = ()


MAX_NOMINATIONS = 4
MAX_NOMINATION_REASON = 120

_TOP_FIELDS = {"schema_version", "run_id", "base_revision", "goal_id", "goal_obligations",
               "dependency_sketch", "evidence_refs", "steps", "keyframe_nominations"}
_STEP_FIELDS = {"id", "skill", "object_id", "destination_id", "arm", "prompt",
                "effects", "evidence_refs", "attempts", "timeout_s"}
_FACT_WRITE_KEYS = {"initial_state", "set", "facts", "world_state", "physical_facts"}


def _fail(path: str, message: str) -> None:
    raise PlanValidationError(f"{path}: {message}")


def _exact_fields(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    extra = set(value) - allowed
    if extra:
        _fail(path, f"unknown fields {sorted(extra)}")


def _finite_number(value: Any, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "expected a number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        _fail(path, f"expected a finite number >= {minimum}")
    return number


def _strings(value: Any, path: str, *, required: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (required and not value):
        _fail(path, "expected a list" + (" with at least one item" if required else ""))
    if any(not isinstance(item, str) or not item for item in value):
        _fail(path, "items must be non-empty strings")
    return tuple(value)


def _contains_forbidden_writes(value: Any) -> None:
    stack = [("proposal", value, 0)]
    visited = 0
    while stack:
        path, current, depth = stack.pop()
        visited += 1
        if visited > 2_000 or depth > 16:
            _fail(path, "proposal structure exceeds bounds")
        if isinstance(current, dict):
            for key, item in current.items():
                if isinstance(key, str) and key.lower() in _FACT_WRITE_KEYS:
                    _fail(f"{path}.{key}", "model proposals cannot write physical or initial state")
                stack.append((f"{path}.{key}", item, depth + 1))
        elif isinstance(current, list):
            for index, item in enumerate(current):
                stack.append((f"{path}[{index}]", item, depth + 1))


def _check_nominations(value: Any, context: PlanningContext) -> tuple[tuple[dict[str, str], ...],
                                                                       tuple[dict[str, str], ...]]:
    """Split optional keyframe nominations into accepted and rejected. Advisory: never raises.

    A nomination is accepted only if it names a frame id this context showed the model, carries a short
    printable reason, and fits the per-proposal bound. A bad nomination cannot sink an otherwise valid plan.
    """
    if value is None:
        return (), ()
    if not context.nominate_keyframes:
        return (), ({"frame_id": "", "error": "keyframe nominations are not enabled for this run"},)
    if not isinstance(value, list):
        return (), ({"frame_id": "", "error": "keyframe_nominations must be a list"},)
    shown = {frame.frame_id for frame in context.observations}
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for item in value:
        frame_id = item.get("frame_id") if isinstance(item, dict) else None
        reason = item.get("reason") if isinstance(item, dict) else None
        label = frame_id if isinstance(frame_id, str) else ""
        if not isinstance(item, dict) or set(item) - {"frame_id", "reason"}:
            rejected.append({"frame_id": label, "error": "expected {frame_id, reason}"})
        elif not isinstance(frame_id, str) or frame_id not in shown:
            rejected.append({"frame_id": label, "error": "frame id was not shown to the planner"})
        elif (not isinstance(reason, str) or not reason.strip() or len(reason) > MAX_NOMINATION_REASON
              or any(ord(character) < 32 for character in reason)):
            rejected.append({"frame_id": frame_id, "error": "reason must be short printable text"})
        elif any(entry["frame_id"] == frame_id for entry in accepted):
            rejected.append({"frame_id": frame_id, "error": "duplicate nomination"})
        elif len(accepted) >= MAX_NOMINATIONS:
            rejected.append({"frame_id": frame_id, "error": "too many nominations"})
        else:
            accepted.append({"frame_id": frame_id, "reason": reason.strip()})
    return tuple(accepted), tuple(rejected)


def validate_proposal(proposal: Any, context: PlanningContext) -> ValidatedPlan:
    """Validate model JSON and compile a short task using trusted goal facts."""

    if not isinstance(proposal, dict):
        _fail("proposal", "expected a JSON object")
    try:
        encoded = json.dumps(proposal, separators=(",", ":"), ensure_ascii=False,
                             allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise PlanValidationError("proposal: must be bounded JSON data") from exc
    if len(encoded) > context.budget.max_proposal_bytes:
        _fail("proposal", "proposal exceeds byte budget")
    _contains_forbidden_writes(proposal)
    _exact_fields(proposal, _TOP_FIELDS, "proposal")
    if proposal.get("schema_version") != 1:
        _fail("proposal.schema_version", "expected version 1")
    if proposal.get("run_id") != context.run_id:
        _fail("proposal.run_id", "proposal belongs to another run")
    revision = proposal.get("base_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision != context.revision:
        _fail("proposal.base_revision", "stale or invalid observation revision")
    if proposal.get("goal_id") != context.goal.goal_id:
        _fail("proposal.goal_id", "proposal does not reference the trusted goal")
    if proposal.get("goal_obligations") == []:
        _fail("proposal.goal_obligations", "proposal changed or weakened trusted goal obligations")
    obligations = _strings(proposal.get("goal_obligations"), "proposal.goal_obligations", required=True)
    if obligations != context.goal.obligations:
        _fail("proposal.goal_obligations", "proposal changed or weakened trusted goal obligations")
    nominations, rejected_nominations = _check_nominations(proposal.get("keyframe_nominations"), context)
    sketch = _strings(proposal.get("dependency_sketch"), "proposal.dependency_sketch")
    if len(sketch) > context.budget.max_sketch_items:
        _fail("proposal.dependency_sketch", "dependency sketch exceeds bound")
    evidence_by_id = {record.evidence_id: record for record in context.evidence}
    frames_by_id = {frame.frame_id: frame for frame in context.observations}

    def check_evidence(ref: str, path: str) -> None:
        if ref in evidence_by_id:
            source_revision = evidence_by_id[ref].revision
            temporal_role = evidence_by_id[ref].temporal_role
        elif ref in frames_by_id:
            source_revision = frames_by_id[ref].revision
            temporal_role = frames_by_id[ref].temporal_role
        else:
            _fail(path, f"unknown evidence reference {ref!r}")
        if source_revision is not None and source_revision != context.revision:
            _fail(path, f"stale evidence reference {ref!r}")
        if temporal_role not in {"current", "historical"}:
            _fail(path, f"invalid evidence temporal role for {ref!r}")

    evidence_refs = _strings(proposal.get("evidence_refs", []), "proposal.evidence_refs")
    for ref in evidence_refs:
        check_evidence(ref, "proposal.evidence_refs")

    steps = proposal.get("steps")
    if not isinstance(steps, list) or not steps:
        _fail("proposal.steps", "expected at least one bounded next step")
    if len(steps) > context.budget.max_steps:
        _fail("proposal.steps", "step count exceeds budget")
    ids: set[str] = set()
    total_seconds = 0.0
    total_attempts = 0
    tree_nodes: list[dict[str, Any]] = []
    selected_evidence = list(evidence_refs)
    for index, step in enumerate(steps):
        path = f"proposal.steps[{index}]"
        if not isinstance(step, dict):
            _fail(path, "expected an object")
        _exact_fields(step, _STEP_FIELDS, path)
        step_id = step.get("id")
        if not isinstance(step_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", step_id):
            _fail(f"{path}.id", "expected a short identifier")
        if step_id in ids:
            _fail(f"{path}.id", "duplicate step id")
        ids.add(step_id)
        skill_id = step.get("skill")
        capability = context.capabilities.skills.get(skill_id) if isinstance(skill_id, str) else None
        if capability is None:
            _fail(f"{path}.skill", f"unregistered skill {skill_id!r}")
        if capability.changes_scene and not capability.effect_templates:
            _fail(f"{path}.skill", "scene-changing skill has no registered effect binding")
        if capability.changes_scene and len(capability.effect_templates) != 1:
            _fail(f"{path}.skill", "only one independently verified effect is supported per invocation")
        arm = step.get("arm")
        if not isinstance(arm, str) or (arm not in context.capabilities.available_arms
                                        and arm not in context.capabilities.arm_groups):
            _fail(f"{path}.arm", f"arm/group {arm!r} is unavailable")
        members = context.capabilities.arm_groups.get(arm, (arm,))
        if arm not in capability.arms or any(member not in context.capabilities.available_arms for member in members):
            _fail(f"{path}.arm", f"skill {skill_id!r} is unavailable on arm/group {arm!r}")
        object_id = step.get("object_id")
        if not isinstance(object_id, str) or object_id not in context.object_labels:
            _fail(f"{path}.object_id", f"unknown object {object_id!r}")
        destination = step.get("destination_id", "")
        if not isinstance(destination, str):
            _fail(f"{path}.destination_id", "expected an object id or empty string")
        if destination and destination not in context.object_labels:
            _fail(f"{path}.destination_id", f"unknown destination object {destination!r}")
        if not destination and any("{destination_id}" in template for template in capability.effect_templates):
            _fail(f"{path}.destination_id", "registered effects require a destination object")
        if capability.changes_scene and not step.get("effects"):
            _fail(f"{path}.effects", "missing required registered effect binding")
        effects = _strings(step.get("effects"), f"{path}.effects")
        try:
            expected_effects = tuple(template.format(object_id=object_id, destination_id=destination)
                                     for template in capability.effect_templates)
        except (KeyError, ValueError) as exc:
            raise PlanValidationError(f"capability {skill_id!r} has an invalid effect template") from exc
        if effects != expected_effects:
            _fail(f"{path}.effects", f"effects must match registered binding {list(expected_effects)}")
        if any(not isinstance(item, str) or not item for item in capability.preconditions):
            _fail(f"capabilities.{skill_id}.preconditions", "registry contains an invalid precondition")
        prompt = step.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > context.budget.max_prompt_chars:
            _fail(f"{path}.prompt", "expected a non-empty bounded proposal explanation")
        timeout = _finite_number(step.get("timeout_s"), f"{path}.timeout_s", minimum=0.1)
        attempts = step.get("attempts")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            _fail(f"{path}.attempts", "expected a positive integer")
        if timeout > min(context.budget.max_seconds, capability.max_timeout_s):
            _fail(f"{path}.timeout_s", "timeout exceeds capability or plan budget")
        if attempts != 1:
            _fail(f"{path}.attempts", "one skill attempt is committed; retries belong to the run lifecycle")
        total_seconds += timeout * attempts
        total_attempts += attempts
        refs = _strings(step.get("evidence_refs", []), f"{path}.evidence_refs")
        for ref in refs:
            check_evidence(ref, f"{path}.evidence_refs")
        selected_evidence.extend(refs)
        operator_arms = context.progress.get("operator_arms", {})
        if effects and isinstance(operator_arms, Mapping) and operator_arms.get(effects[0], arm) != arm:
            # The operator accepted this arm for this effect; the planner may not reassign it.
            _fail(f"{path}.arm", f"operator assigned {effects[0]!r} to arm {operator_arms[effects[0]]!r}")
        if effects and effects[0] in set(context.progress.get("completed_effects", ())):
            _fail(f"{path}.effects", "proposed effect is already completed according to trusted progress")
        try:
            trusted_prompt = capability.prompt_template.format(
                skill_id=skill_id, object_id=object_id, destination_id=destination,
                object_label=context.object_labels[object_id],
                destination_label=context.object_labels.get(destination, ""),
            )
        except (KeyError, ValueError) as exc:
            raise PlanValidationError(f"capability {skill_id!r} has an invalid prompt template") from exc
        if any(ord(character) < 32 for character in trusted_prompt):
            _fail(f"capabilities.{skill_id}.prompt_template", "rendered prompt contains control characters")
        if capability.preconditions:
            tree_nodes.extend({"check": predicate} for predicate in capability.preconditions)
        tree_nodes.append({"skill": {"arm": arm, "name": skill_id, "prompt": trusted_prompt,
                                     "target": object_id, "success": effects[0] if effects else None,
                                     "timeout_s": timeout,
                                     "params": {"object_id": object_id, "destination_id": destination,
                                                "resources": capability.resources}}})
    if total_seconds > context.budget.max_seconds:
        _fail("proposal.steps", "aggregate time exceeds plan budget")
    if total_attempts > context.budget.max_attempts:
        _fail("proposal.steps", "aggregate attempts exceed plan budget")

    # The executable task's goal always comes from GoalSpec, never proposal prose.
    task_data = {
        "name": f"plan_{context.goal.goal_id}",
        "goal": tree_nodes[-1]["skill"]["success"] if "skill" in tree_nodes[-1] else context.goal.obligations[0],
        "description": context.goal.text,
        "tree": tree_nodes[0] if len(tree_nodes) == 1 else {"sequence": tree_nodes},
    }
    try:
        available_targets = set(context.capabilities.available_arms)
        available_targets.update(
            group for group, members in context.capabilities.arm_groups.items()
            if members and all(member in context.capabilities.available_arms for member in members)
        )
        task = parse_task(task_data, arms=available_targets,
                          skills=context.capabilities.skills.keys())
    except (TaskSpecError, TypeError, ValueError) as exc:
        raise PlanValidationError(f"rendered task: {exc}") from exc
    return ValidatedPlan(context.run_id, revision, context.goal.goal_id, sketch, task,
                         tuple(step["id"] for step in steps), tuple(dict.fromkeys(selected_evidence)),
                         nominations, rejected_nominations)


def parse_proposal_json(text: str, max_bytes: int = 16_384) -> dict[str, Any]:
    """Parse one bounded JSON proposal, accepting a single fenced JSON block."""

    if not isinstance(text, str):
        _fail("proposal", "response is missing or exceeds byte budget")
    try:
        encoded_size = len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise PlanValidationError("proposal: response contains invalid Unicode") from exc
    if encoded_size > max_bytes:
        _fail("proposal", "response is missing or exceeds byte budget")
    match = re.fullmatch(r"\s*```(?:json)?\s*\n?(.*?)\n?```\s*", text, re.DOTALL | re.IGNORECASE)
    payload = match.group(1) if match else text
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise PlanValidationError(f"proposal: malformed JSON: {exc}") from exc
    if not isinstance(value, dict):
        _fail("proposal", "expected one JSON object")
    return value
