"""Deterministic, image-producing replay fixtures for the V3 memory demo.

These components provide saved symbolic scenarios and schematic images. They do not
call a VLM, infer physical facts from pixels, or control hardware.
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
import uuid
import zlib
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from liminal.core.types import Skill, SkillResult, Status
from liminal.orchestration.plan_validation import (
    CapabilityRegistry,
    GoalSpec,
    SkillCapability,
)
from liminal.perception.evidence import Truth


@dataclass(frozen=True)
class ReplayScenario:
    name: str
    goal: GoalSpec
    objects: dict[str, str]
    obligations: dict[str, dict[str, str]]
    registry: CapabilityRegistry
    reference_frames: dict[str, str] = field(default_factory=dict)
    initial_locations: dict[str, str] = field(default_factory=dict)
    initially_occluded: tuple[str, ...] = ()
    # Scheduled external perturbations (a person moves something), keyed by the number of scene
    # actions (robot placements that changed the scene) after which they fire.
    events: tuple[ScheduledEvent, ...] = ()
    # V5 cooperative state (all empty for the object-placement scenarios): observed scene
    # predicates beyond `{object}_in_{destination}`, the skills that touch them, which arm
    # supports which state, and scheduled physical faults. See ``_load_state_spec``.
    states: dict[str, bool] = field(default_factory=dict)
    state_skills: dict[str, StateSkill] = field(default_factory=dict)
    supports: dict[str, SupportSpec] = field(default_factory=dict)
    faults: tuple[StateFault, ...] = ()


@dataclass(frozen=True)
class StateSkill:
    """Scene semantics of a scenario-declared skill; its effects come from its capability."""

    duration_s: float = 0.0
    requires: tuple[str, ...] = ()  # state predicates that must hold when the skill completes


@dataclass(frozen=True)
class SupportSpec:
    arm: str  # the arm whose hold keeps the state true
    release_closes: bool = True  # releasing/homing that arm lets the state fall (lid closes)


@dataclass(frozen=True)
class StateFault:
    """A scheduled physical fault on a state predicate.

    ``slip_during``: ``at_s`` into the ``on_attempt``-th run of ``skill`` the state goes false,
    unless ``unless_supported_by`` names an arm that is holding at that moment.
    ``support_lost``: ``after_s`` seconds after the state became true it falls, once.
    """

    kind: str
    state: str
    skill: str = ""
    on_attempt: int = 1
    at_s: float = 0.0
    after_s: float = 0.0
    unless_supported_by: str = ""


@dataclass(frozen=True)
class ScheduledEvent:
    after_actions: int
    kind: str  # "relocate" | "occlude" | "reveal"
    object_id: str
    location: str = "table"


@dataclass(frozen=True)
class SceneImage:
    frame_id: str
    camera: str
    captured_at: float
    data: bytes
    media_type: str
    event: str

    @property
    def timestamp(self) -> float:
        """Wall-clock capture time, as an alias for ``captured_at``."""
        return self.captured_at


@dataclass(frozen=True)
class SceneSnapshot:
    revision: int
    facts: dict[str, Truth]
    bindings: dict[str, dict[str, str]]
    images: tuple[SceneImage, ...]
    captured_mono_s: float


def load_scenario(path: str | Path) -> ReplayScenario:
    """Load and validate one authored replay fixture from YAML."""
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("replay scenario must be a YAML object")
    allowed = {"name", "goal", "objects", "obligations", "capabilities", "reference_frames", "initial_locations",
               "initially_occluded", "events", "state"}
    extra = set(raw) - allowed
    if extra:
        raise ValueError(f"unknown scenario fields: {sorted(extra)}")
    name = _required_string(raw, "name", "scenario")
    goal_data = raw.get("goal")
    if not isinstance(goal_data, dict) or set(goal_data) != {"id", "text", "obligations"}:
        raise ValueError("goal must contain exactly id, text and obligations")
    goal_ids = _string_list(goal_data.get("obligations"), "goal.obligations")
    goal = GoalSpec(
        _required_string(goal_data, "id", "goal"),
        _required_string(goal_data, "text", "goal"),
        tuple(goal_ids),
    )
    objects_raw = raw.get("objects")
    if not isinstance(objects_raw, dict) or not objects_raw:
        raise ValueError("objects must map object IDs to display labels")
    objects = {str(key): _nonempty_string(value, f"objects.{key}") for key, value in objects_raw.items()}
    if any(not key for key in objects):
        raise ValueError("object IDs must be non-empty")

    obligations_raw = raw.get("obligations")
    if not isinstance(obligations_raw, dict) or set(obligations_raw) != set(goal.obligations):
        raise ValueError("obligations must define exactly the trusted goal predicates")
    obligations: dict[str, dict[str, str]] = {}
    for predicate, binding in obligations_raw.items():
        if not isinstance(binding, dict) or set(binding) != {"object_id", "destination_id"}:
            raise ValueError(f"obligations.{predicate} must bind object_id and destination_id")
        object_id = _required_string(binding, "object_id", f"obligations.{predicate}")
        destination_id = _required_string(binding, "destination_id", f"obligations.{predicate}")
        if object_id not in objects or destination_id not in objects or object_id == destination_id:
            raise ValueError(f"obligations.{predicate} references unknown or identical object IDs")
        obligations[predicate] = {"object_id": object_id, "destination_id": destination_id}

    caps_raw = raw.get("capabilities")
    if not isinstance(caps_raw, dict):
        raise ValueError("capabilities must be a YAML object")
    available_arms = tuple(_string_list(caps_raw.get("available_arms"), "capabilities.available_arms"))
    if not available_arms:
        raise ValueError("capabilities.available_arms must not be empty")
    groups_raw = caps_raw.get("arm_groups", {})
    if not isinstance(groups_raw, dict):
        raise ValueError("capabilities.arm_groups must be an object")
    arm_groups = {str(group): tuple(_string_list(members, f"capabilities.arm_groups.{group}"))
                  for group, members in groups_raw.items()}
    skills_raw = caps_raw.get("skills")
    if not isinstance(skills_raw, dict) or not skills_raw:
        raise ValueError("capabilities.skills must be a non-empty object")
    skills: dict[str, SkillCapability] = {}
    skill_fields = {
        "arms", "effect_templates", "prompt_template", "max_timeout_s", "max_attempts",
        "resources", "changes_scene", "preconditions",
    }
    for skill_id, values in skills_raw.items():
        if not isinstance(skill_id, str) or not skill_id or not isinstance(values, dict):
            raise ValueError("skill capabilities must map non-empty IDs to objects")
        unknown = set(values) - skill_fields
        if unknown:
            raise ValueError(f"capabilities.skills.{skill_id}: unknown fields {sorted(unknown)}")
        skills[skill_id] = SkillCapability(
            skill_id=skill_id,
            arms=tuple(_string_list(values.get("arms"), f"capabilities.skills.{skill_id}.arms")),
            effect_templates=tuple(
                _string_list(values.get("effect_templates", []), f"capabilities.skills.{skill_id}.effect_templates")
            ),
            prompt_template=values.get("prompt_template", "{skill_id} {object_label} {destination_label}"),
            max_timeout_s=values.get("max_timeout_s", 15),
            max_attempts=values.get("max_attempts", 1),
            resources=tuple(_string_list(values.get("resources", []), f"capabilities.skills.{skill_id}.resources")),
            changes_scene=values.get("changes_scene", True),
            preconditions=tuple(
                _string_list(values.get("preconditions", []), f"capabilities.skills.{skill_id}.preconditions")
            ),
        )
    registry = CapabilityRegistry(
        skills=skills,
        available_arms=available_arms,
        arm_groups=arm_groups,
        version=str(caps_raw.get("version", "replay-v1")),
    )

    refs_raw = raw.get("reference_frames", {})
    if not isinstance(refs_raw, dict):
        raise ValueError("reference_frames must map reference IDs to known object IDs")
    reference_frames = {str(frame_id): _nonempty_string(object_id, f"reference_frames.{frame_id}")
                        for frame_id, object_id in refs_raw.items()}
    if any(object_id not in objects for object_id in reference_frames.values()):
        raise ValueError("reference_frames references an unknown object")
    locations_raw = raw.get("initial_locations", {})
    if not isinstance(locations_raw, dict):
        raise ValueError("initial_locations must map object IDs to location IDs")
    initial_locations = {str(object_id): _nonempty_string(location, f"initial_locations.{object_id}")
                         for object_id, location in locations_raw.items()}
    if set(initial_locations) - set(objects) or any(location not in objects and location != "table"
                                                    for location in initial_locations.values()):
        raise ValueError("initial_locations references an unknown object or location")
    occluded = tuple(_string_list(raw.get("initially_occluded", []), "initially_occluded"))
    tracked = {binding["object_id"] for binding in obligations.values()}
    if any(object_id not in tracked for object_id in occluded):
        raise ValueError("initially_occluded must name objects that appear in an obligation")
    events = _load_events(raw.get("events", []), tracked, objects)
    states, state_skills, supports, faults = _load_state_spec(raw.get("state"), obligations, registry)
    return ReplayScenario(name, goal, objects, obligations, registry, reference_frames, initial_locations, occluded,
                          events, states, state_skills, supports, faults)


def _load_state_spec(raw: Any, obligations: Mapping[str, Any], registry: CapabilityRegistry,
                     ) -> tuple[dict[str, bool], dict[str, StateSkill], dict[str, SupportSpec], tuple[StateFault, ...]]:
    """The optional ``state:`` block (V5 cooperative scenarios)::

        state:
          predicates: {lid_open: false}          # observed scene state and its initial value
          skills:                                # scene semantics of capabilities.skills entries
            open_lid: {duration_s: 0.1}          #   effect `lid_open` (a template without braces) sets it
            insert: {duration_s: 0.5, requires: [lid_open]}   # effect {object_id}_in_{destination_id}
          support: {lid_open: {arm: left, release_closes: true}}
          faults:
            - {kind: slip_during, skill: insert, state: lid_open, at_s: 0.2, on_attempt: 1,
               unless_supported_by: left}        # omit unless_supported_by: slips even when held
            - {kind: support_lost, state: lid_open, after_s: 1.5}
    """
    if raw is None:
        return {}, {}, {}, ()
    if not isinstance(raw, dict) or set(raw) - {"predicates", "skills", "support", "faults"}:
        raise ValueError("state must be an object with predicates, skills, support and faults")
    preds = raw.get("predicates")
    if not isinstance(preds, dict) or not preds:
        raise ValueError("state.predicates must map predicate names to initial booleans")
    states = {}
    for key, value in preds.items():
        if not isinstance(key, str) or not key or not isinstance(value, bool) or key in obligations:
            raise ValueError(f"state.predicates.{key}: needs a unique name and a boolean initial value")
        states[key] = value
    skills_raw = raw.get("skills", {})
    if not isinstance(skills_raw, dict):
        raise ValueError("state.skills must be an object")
    state_skills = {}
    for skill_id, values in skills_raw.items():
        where = f"state.skills.{skill_id}"
        if skill_id not in registry.skills:
            raise ValueError(f"{where} must also be declared in capabilities.skills")
        if not isinstance(values, dict) or set(values) - {"duration_s", "requires"}:
            raise ValueError(f"{where} takes duration_s and requires")
        duration = values.get("duration_s", 0.0)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
            raise ValueError(f"{where}.duration_s must be a non-negative number")
        requires = tuple(_string_list(values.get("requires", []), f"{where}.requires"))
        if any(r not in states for r in requires):
            raise ValueError(f"{where}.requires names an unknown state predicate")
        for template in registry.skills[skill_id].effect_templates:
            if "{" not in template and template not in states:
                raise ValueError(f"{where}: effect {template!r} is neither a state predicate nor a template")
        state_skills[skill_id] = StateSkill(float(duration), requires)
    supports = {}
    for key, values in (raw.get("support") or {}).items():
        if key not in states or not isinstance(values, dict) or set(values) - {"arm", "release_closes"} \
                or values.get("arm") not in registry.available_arms:
            raise ValueError(f"state.support.{key} needs a known state and an available arm")
        supports[key] = SupportSpec(values["arm"], bool(values.get("release_closes", True)))
    faults = []
    for index, item in enumerate(raw.get("faults") or []):
        where = f"state.faults[{index}]"
        if not isinstance(item, dict) or set(item) - {"kind", "state", "skill", "on_attempt", "at_s", "after_s",
                                                       "unless_supported_by"}:
            raise ValueError(f"{where} has unknown fields")
        kind = item.get("kind")
        if kind not in ("slip_during", "support_lost") or item.get("state") not in states:
            raise ValueError(f"{where} needs kind slip_during|support_lost and a known state")
        fault = StateFault(kind, item["state"], str(item.get("skill", "")), int(item.get("on_attempt", 1)),
                           float(item.get("at_s", 0.0)), float(item.get("after_s", 0.0)),
                           str(item.get("unless_supported_by", "")))
        if kind == "slip_during" and fault.skill not in state_skills:
            raise ValueError(f"{where}.skill must be a state skill")
        if fault.unless_supported_by and fault.unless_supported_by not in registry.available_arms:
            raise ValueError(f"{where}.unless_supported_by is not an available arm")
        faults.append(fault)
    return states, state_skills, supports, tuple(faults)


def _load_events(raw: Any, tracked: set[str], objects: Mapping[str, str]) -> tuple[ScheduledEvent, ...]:
    if not isinstance(raw, list):
        raise ValueError("events must be a list")
    events = []
    for index, item in enumerate(raw):
        where = f"events[{index}]"
        if not isinstance(item, dict) or not {"after_actions", "kind", "object_id"} <= set(item) \
                or set(item) - {"after_actions", "kind", "object_id", "location"}:
            raise ValueError(f"{where} needs after_actions, kind, object_id (and location for relocate)")
        after = item["after_actions"]
        if isinstance(after, bool) or not isinstance(after, int) or after < 1:
            raise ValueError(f"{where}.after_actions must be a positive integer")
        kind = item["kind"]
        if kind not in ("relocate", "occlude", "reveal"):
            raise ValueError(f"{where}.kind must be relocate, occlude or reveal")
        object_id = _nonempty_string(item["object_id"], f"{where}.object_id")
        if object_id not in tracked:
            raise ValueError(f"{where}.object_id must appear in an obligation")
        location = _nonempty_string(item.get("location", "table"), f"{where}.location")
        if location != "table" and (location not in objects or location == object_id):
            raise ValueError(f"{where}.location is unknown")
        events.append(ScheduledEvent(after, kind, object_id, location))
    return tuple(events)


class ReplayScene:
    """Persistent symbolic scene with deterministic, explicitly schematic PNG frames."""

    def __init__(self, scenario: ReplayScenario, state_path: str | Path):
        self.scenario = scenario
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._initial_locations = {
            object_id: scenario.initial_locations.get(object_id, "table")
            for object_id in scenario.objects
            if any(binding["object_id"] == object_id for binding in scenario.obligations.values())
        }
        self._reference_created_at = time.time()
        if self.state_path.exists():
            self._load_state()
        else:
            self.revision = 0
            self.locations = dict(self._initial_locations)
            self.occluded = set(scenario.initially_occluded)
            self.actions = 0
            self.events_fired = 0
            self.states = dict(scenario.states)
            self.faults_fired: set[int] = set()
            self._save_state()
        # In-memory only: skill run counters, fault outcomes and when each state last became true.
        self.runs: dict[str, int] = {}
        self.fault_log: list[dict[str, Any]] = []
        now = time.monotonic()
        self._true_since = {key: now for key, value in self.states.items() if value}

    def _load_state(self) -> None:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read replay scene state {self.state_path}") from exc
        if data.get("scenario") != self.scenario.name:
            raise ValueError("replay scene state belongs to a different scenario")
        revision = data.get("revision")
        locations = data.get("locations")
        if (isinstance(revision, bool) or not isinstance(revision, int) or revision < 0
                or not isinstance(locations, dict)):
            raise ValueError("replay scene state is malformed")
        if set(locations) != set(self._initial_locations):
            raise ValueError("replay scene object set differs from the saved state")
        for location in locations.values():
            if not isinstance(location, str) or (location != "table" and location not in self.scenario.objects):
                raise ValueError("replay scene contains an unknown object location")
        created_at = data.get("reference_created_at", self._reference_created_at)
        if isinstance(created_at, bool) or not isinstance(created_at, (int, float)):
            raise ValueError("replay reference timestamp is malformed")
        occluded = data.get("occluded", list(self.scenario.initially_occluded))
        if not isinstance(occluded, list) or not set(occluded) <= set(self._initial_locations):
            raise ValueError("replay scene occlusion state is malformed")
        self.actions = int(data.get("actions", 0))
        self.events_fired = int(data.get("events_fired", 0))
        self.revision = revision
        self.occluded = set(occluded)
        self.locations = dict(locations)
        self._reference_created_at = float(created_at)
        states = data.get("states", dict(self.scenario.states))
        if (not isinstance(states, dict) or set(states) != set(self.scenario.states)
                or any(not isinstance(v, bool) for v in states.values())):
            raise ValueError("replay scene state predicates differ from the saved state")
        self.states = dict(states)
        self.faults_fired = {int(i) for i in data.get("faults_fired", [])}

    def _save_state(self) -> None:
        data = {
            "schema_version": 1,
            "scenario": self.scenario.name,
            "revision": self.revision,
            "locations": self.locations,
            "occluded": sorted(self.occluded),
            "actions": self.actions,
            "events_fired": self.events_fired,
            "reference_created_at": self._reference_created_at,
        }
        if self.scenario.states:  # object-only scenarios keep their exact saved format
            data["states"] = self.states
            data["faults_fired"] = sorted(self.faults_fired)
        temp_path = self.state_path.with_name(self.state_path.name + ".tmp")
        temp_path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.state_path)

    def capture(self) -> SceneSnapshot:
        self._tick()
        captured_mono = time.monotonic()
        captured_at = time.time()
        images: list[SceneImage] = []
        for reference_id, object_id in self.scenario.reference_frames.items():
            reference_locations = dict(self._initial_locations)
            images.append(SceneImage(
                frame_id=f"reference:{self.scenario.name}:{reference_id}",
                camera="scene",
                captured_at=self._reference_created_at,
                data=_render_png(self.scenario, reference_locations, marked=object_id),
                media_type="image/png",
                event=f"goal_reference:{object_id}",
            ))
        images.append(SceneImage(
            frame_id=f"scene:{self.scenario.name}:{uuid.uuid4().hex}",
            camera="scene",
            captured_at=captured_at,
            data=_render_png(self.scenario, self.locations, hidden=self.occluded, states=self.states),
            media_type="image/png",
            event="schematic_current_scene",
        ))
        facts: dict[str, Truth] = {}
        bindings: dict[str, dict[str, str]] = {}
        for predicate, binding in self.scenario.obligations.items():
            bindings[predicate] = dict(binding)
            if binding["object_id"] in self.occluded:
                # A hidden object is not evidence of absence: stay UNKNOWN, never FALSE.
                facts[predicate] = Truth.UNKNOWN
            else:
                facts[predicate] = (
                    Truth.TRUE if self.locations.get(binding["object_id"]) == binding["destination_id"]
                    else Truth.FALSE
                )
        for state, value in self.states.items():
            facts[state] = Truth.TRUE if value else Truth.FALSE
            bindings[state] = {}
        return SceneSnapshot(self.revision, facts, bindings, tuple(images), captured_mono)

    # -- V5 cooperative state ---------------------------------------------------------------

    def begin_state_skill(self, skill: Skill) -> StateFault | None:
        """Count a run of a state skill and return the slip fault scheduled for this run, if any."""
        run = self.runs[skill.name] = self.runs.get(skill.name, 0) + 1
        return next((f for f in self.scenario.faults
                     if f.kind == "slip_during" and f.skill == skill.name and f.on_attempt == run), None)

    def slip(self, fault: StateFault, supported_arms: Collection[str]) -> bool:
        """Apply a slip fault now; an arm that is holding the support suppresses it."""
        if fault.unless_supported_by and fault.unless_supported_by in supported_arms:
            self.fault_log.append({"fault": fault.kind, "state": fault.state, "outcome": "suppressed"})
            return False
        self._set_state(fault.state, False)
        self.fault_log.append({"fault": fault.kind, "state": fault.state, "outcome": "fired"})
        return True

    def complete_state_skill(self, skill: Skill) -> bool:
        """Apply a state skill's effects if its physical preconditions hold; False = scene unchanged."""
        self._tick()
        decl = self.scenario.state_skills[skill.name]
        capability = self.scenario.registry.skills[skill.name]
        if skill.arm not in capability.arms or skill.arm not in self.scenario.registry.available_arms:
            raise ValueError(f"skill {skill.name!r} is unavailable on arm {skill.arm!r}")
        unmet = [r for r in decl.requires if not self.states[r]]
        if unmet:
            self.fault_log.append({"skill": skill.name, "outcome": "blocked", "unmet": unmet})
            return False
        for template in capability.effect_templates:
            if "{" in template:
                self.apply(skill)
            else:
                self._set_state(template, True)
        return True

    def support_released(self, arm: str) -> None:
        """The support arm let go (release or home): states it kept up fall."""
        for state, spec in self.scenario.supports.items():
            if spec.arm == arm and spec.release_closes and self.states[state]:
                self._set_state(state, False)
                self.fault_log.append({"state": state, "outcome": "released"})

    def _set_state(self, state: str, value: bool) -> None:
        if self.states[state] == value:
            return
        self.states[state] = value
        if value:
            self._true_since[state] = time.monotonic()
        else:
            self._true_since.pop(state, None)
        self.revision += 1
        self._save_state()

    def _tick(self) -> None:
        """Lazily fire time-based faults: the world moves whether or not anyone is looking."""
        now = time.monotonic()
        for index, fault in enumerate(self.scenario.faults):
            since = self._true_since.get(fault.state)
            if (fault.kind == "support_lost" and index not in self.faults_fired and since is not None
                    and now - since >= fault.after_s):
                self.faults_fired.add(index)
                self._set_state(fault.state, False)
                self.fault_log.append({"fault": fault.kind, "state": fault.state, "outcome": "fired"})

    def apply(self, skill: Skill) -> int:
        """Apply the action encoded by scenario bindings, ignoring `skill.success`."""
        capability = self.scenario.registry.skills.get(skill.name)
        if capability is None or not capability.changes_scene:
            raise ValueError(f"skill is not a registered replay scene action: {skill.name!r}")
        if skill.arm not in capability.arms or skill.arm not in self.scenario.registry.available_arms:
            raise ValueError(f"skill {skill.name!r} is unavailable on arm {skill.arm!r}")
        params = skill.params
        object_id = params.get("object_id")
        destination_id = params.get("destination_id")
        if not isinstance(object_id, str) or not isinstance(destination_id, str):
            raise ValueError("replay action requires object_id and destination_id skill parameters")
        if skill.target != object_id:
            raise ValueError("skill target and object_id parameter do not match")
        predicate = next((
            name for name, binding in self.scenario.obligations.items()
            if binding == {"object_id": object_id, "destination_id": destination_id}
        ), None)
        if predicate is None:
            raise ValueError("object/destination pair is not a scenario obligation")
        expected_effects = tuple(
            template.format(object_id=object_id, destination_id=destination_id)
            for template in capability.effect_templates
        )
        if predicate not in expected_effects:
            raise ValueError("registered skill does not support the scenario object/destination effect")
        changed = False
        if self.locations[object_id] != destination_id:
            self.locations[object_id] = destination_id
            changed = True
        if object_id in self.occluded:
            # The arm handled the object, so it is no longer hidden from the camera.
            self.occluded.discard(object_id)
            changed = True
        if changed:
            self.revision += 1
            self.actions += 1
            self._save_state()
        return self.revision

    def fire_due_events(self) -> list[ScheduledEvent]:
        """Apply scheduled perturbations whose action count is reached (once each, persisted).

        Called by the backend right after a scene action, i.e. while the agent is "thinking" or
        verifying, so the next observation sees a world that moved on its own.
        """
        fired = []
        pending = self.scenario.events[self.events_fired:]
        for event in pending:
            if event.after_actions > self.actions:
                break
            if event.kind == "relocate":
                self.relocate(event.object_id, event.location)
            elif event.kind == "occlude":
                self.occlude(event.object_id)
            else:
                self.reveal(event.object_id)
            self.events_fired += 1
            self._save_state()
            fired.append(event)
        return fired

    def occlude(self, object_id: str) -> int:
        """Hide a tracked object from the camera; its predicates become UNKNOWN."""
        self._require_tracked(object_id)
        if object_id not in self.occluded:
            self.occluded.add(object_id)
            self.revision += 1
            self._save_state()
        return self.revision

    def reveal(self, object_id: str) -> int:
        self._require_tracked(object_id)
        if object_id in self.occluded:
            self.occluded.discard(object_id)
            self.revision += 1
            self._save_state()
        return self.revision

    def relocate(self, object_id: str, location: str) -> int:
        """External perturbation (a person moves an object); not a robot action."""
        self._require_tracked(object_id)
        if location != "table" and (location not in self.scenario.objects or location == object_id):
            raise ValueError(f"unknown replay location {location!r}")
        if self.locations[object_id] != location:
            self.locations[object_id] = location
            self.revision += 1
            self._save_state()
        return self.revision

    def _require_tracked(self, object_id: str) -> None:
        if object_id not in self._initial_locations:
            raise ValueError(f"object {object_id!r} is not tracked by this replay scene")


@dataclass(frozen=True)
class ReplayCall:
    skill_id: str
    arm: str
    object_id: str | None
    destination_id: str | None
    result: Status
    revision_before: int
    revision_after: int


@dataclass
class ReplaySession:
    """One arm's policy-session stand-in (AGENTS.md rule 6): its own call history and cancel count.

    The real FLUX session keeps an observation/action-chunk queue per skill target; a replay
    session keeps the equivalent bookkeeping so a test can prove the arms never share it.
    """

    arm: str
    calls: list[ReplayCall] = field(default_factory=list)
    cancelled: int = 0


class ReplayBackend:
    """A scenario-bound SkillBackend with failure and false-success injection.

    Scenario-declared state skills (``state.skills``) take their ``duration_s`` of simulated
    time, so a cancel mid-skill leaves the scene untouched, and may run into scheduled slip
    faults. Per-arm bookkeeping lives in ``sessions``; ``calls`` is the global history.
    """

    def __init__(self, scene: ReplayScene, *, fail_first: bool = False, lie_first: bool = False):
        self.scene = scene
        self.fail_first = fail_first
        self.lie_first = lie_first
        self.calls: list[ReplayCall] = []
        self.sessions: dict[str, ReplaySession] = {}

    def session(self, arm: str) -> ReplaySession:
        return self.sessions.setdefault(arm, ReplaySession(arm))

    async def execute(self, skill: Skill, ctx: Any) -> SkillResult:
        before = self.scene.revision
        call_index = len(self.calls) + 1
        object_id = skill.params.get("object_id")
        destination_id = skill.params.get("destination_id")
        session = self.session(skill.arm)
        if self.fail_first and call_index == 1:
            result = Status.FAILURE
            message = "injected replay failure; physical scene unchanged"
        elif self.lie_first and call_index == (2 if self.fail_first else 1):
            result = Status.SUCCESS
            message = "injected backend success claim; physical scene unchanged"
        elif skill.name in self.scene.scenario.state_skills:
            try:
                result, message = await self._run_state_skill(skill, ctx)
            except asyncio.CancelledError:
                session.cancelled += 1  # a cancelled skill leaves no effect and no stale session state
                raise
        else:
            self.scene.apply(skill)
            result = Status.SUCCESS
            message = "authored replay action applied to persistent scene"
        call = ReplayCall(skill.name, skill.arm, object_id, destination_id, result, before, self.scene.revision)
        self.calls.append(call)
        session.calls.append(call)
        return SkillResult(result, skill, message, 0.0)

    async def _run_state_skill(self, skill: Skill, ctx: Any) -> tuple[Status, str]:
        decl = self.scene.scenario.state_skills[skill.name]
        fault = self.scene.begin_state_skill(skill)
        elapsed = 0.0
        if fault is not None:
            elapsed = min(fault.at_s, decl.duration_s)
            await asyncio.sleep(elapsed)
            holding = {name for name, arm in getattr(ctx, "arms", {}).items()
                       if getattr(arm, "holding", None) is not None}
            self.scene.slip(fault, holding)
        await asyncio.sleep(decl.duration_s - elapsed)
        if self.scene.complete_state_skill(skill):
            return Status.SUCCESS, "authored replay state skill applied to persistent scene"
        # A policy cannot see the physics: it reports done, the verifier is what says no.
        return Status.SUCCESS, "replay policy reports done; a precondition failed, scene unchanged"


class ReplayModel:
    """Deterministic authored proposal fixture, not a model or VLM reasoning path."""

    def __init__(self, scenario: ReplayScenario, scene: ReplayScene | None = None):
        self.scenario = scenario
        self.scene = scene  # when given, scheduled scenario events fire "while the model thinks"
        self.calls = 0

    def complete(self, system: str, user: str, images: Sequence[bytes] = ()) -> str:
        del system, images
        return self._reply(user)

    def complete_context(self, system: str, user: str, frames: Sequence[Any]) -> str:
        del system, frames
        return self._reply(user)

    def _reply(self, user: str) -> str:
        reply = self._proposal(user)
        if self.scene is not None:
            # The proposal was grounded in the scene as it was; if a scheduled event changes the scene
            # now, the episode must notice the revision moved and drop this plan.
            self.scene.fire_due_events()
        return reply

    def _proposal(self, user: str) -> str:
        self.calls += 1
        payload = json.loads(user)
        trusted_goal = payload["trusted_goal"]
        progress = payload.get("progress", {})
        completed = set(progress.get("completed_effects", ())) if isinstance(progress, dict) else set()
        pending = progress.get("pending_effects") if isinstance(progress, dict) else None
        if pending is None:
            pending = [name for name in trusted_goal["obligations"] if name not in completed]
        if isinstance(pending, dict):
            pending_items = [dict(predicate=predicate, **binding) for predicate, binding in pending.items()]
        else:
            pending_items = []
            for item in pending:
                if isinstance(item, str):
                    expected = self.scenario.obligations.get(item)
                    if expected:
                        pending_items.append({"predicate": item, **expected})
                elif isinstance(item, dict):
                    pending_items.append(item)
        chosen: tuple[str, str, str] | None = None
        for item in pending_items:
            if not isinstance(item, dict):
                continue
            predicate = item.get("predicate")
            binding = self.scenario.obligations.get(predicate)
            if (not binding or item.get("object_id") != binding["object_id"]
                    or item.get("destination_id") != binding["destination_id"]):
                continue
            if predicate in completed:
                continue
            chosen = predicate, binding["object_id"], binding["destination_id"]
            break
        if chosen is None:
            raise ValueError("ReplayModel requires a pending, scenario-grounded effect")
        predicate, object_id, destination_id = chosen
        capability_data = payload.get("runtime_capabilities", [])
        selected_capability: dict[str, Any] | None = None
        skill_id = ""
        for capability in capability_data:
            if not isinstance(capability, dict) or not capability.get("changes_scene", True):
                continue
            effects = capability.get("effect_templates", [])
            expected = [template.format(object_id=object_id, destination_id=destination_id)
                        for template in effects]
            if predicate in expected:
                selected_capability = capability
                skill_id = capability["skill_id"]
                break
        if selected_capability is None:
            raise ValueError(f"no runtime capability supports scenario effect {predicate!r}")
        arms = selected_capability.get("arms", [])
        available = set(payload.get("available_arms", []))
        assigned = (payload.get("progress") or {}).get("operator_arms", {}).get(predicate)
        arm = assigned if assigned in arms and assigned in available else \
            next((candidate for candidate in arms if candidate in available), None)
        if arm is None:
            raise ValueError(f"capability {skill_id!r} has no available replay arm")
        labels = payload.get("objects", {})
        frame_refs = [item["frame_id"] for item in payload.get("frames", [])
                      if isinstance(item, dict) and item.get("frame_id")]
        timeout_s = min(10.0, float(selected_capability.get("max_timeout_s", 15.0)))
        proposal = {
            "schema_version": 1,
            "run_id": payload["run_id"],
            "base_revision": payload["revision"],
            "goal_id": trusted_goal["id"],
            "goal_obligations": list(trusted_goal["obligations"]),
            "dependency_sketch": [f"{object_id} to {destination_id}"],
            "evidence_refs": frame_refs[:2],
            "steps": [{
                "id": f"place-{object_id}",
                "skill": skill_id,
                "object_id": object_id,
                "destination_id": destination_id,
                "arm": arm,
                "prompt": (f"Place {labels.get(object_id, object_id)} inside "
                           f"{labels.get(destination_id, destination_id)}."),
                "effects": [predicate],
                "evidence_refs": frame_refs[:2],
                "timeout_s": timeout_s,
                "attempts": 1,
            }],
        }
        return json.dumps(proposal, separators=(",", ":"))


class ReplayObserver:
    """Turns scene ground truth into evidence-backed observations in a WorldState.

    This is the only writer of the scenario's physical predicates. It reads the scene, never
    the backend's reported status, so a backend that claims success without moving anything
    is contradicted by the next refresh. Hidden objects yield UNKNOWN, not FALSE.
    """

    OBSERVER_ID = "replay_scene"
    VERSION = "1"

    def __init__(self, world: Any, scene: ReplayScene, *, run_id: str = "", attempt_id: str = "",
                 ttl_s: float = 5.0, bound: bool = True):
        from liminal.perception.evidence import PredicateSpec

        self.world = world
        self.scene = scene
        self.run_id = run_id
        self.attempt_id = attempt_id
        # bound=False writes facts without object bindings: what authored trees (Check, holds) read.
        self.bound = bound
        self.last_snapshot: SceneSnapshot | None = None
        self.refresh_count = 0
        self._frame_seq = 0
        self._observe_count = 0
        self._epoch = uuid.uuid4().hex[:8]
        predicates = set(scene.scenario.obligations) | set(scene.scenario.states)
        for predicate in sorted(predicates):
            world.register_predicate(PredicateSpec(predicate, ttl_s=ttl_s))
        self._writer = world.register_observer(self.OBSERVER_ID, self.VERSION, predicates)

    def bind(self, run_id: str, attempt_id: str) -> None:
        self.run_id, self.attempt_id = run_id, attempt_id

    def make_refresh(self, run_id: str, attempt_id: str) -> Callable[[], Awaitable[None]]:
        """Return an AttemptScope.refresh callable bound to one run/attempt."""
        async def refresh() -> None:
            await self.observe(run_id, attempt_id)
        return refresh

    async def refresh(self) -> None:
        """AttemptScope.refresh / ctx.observe: ids come from ``bind``, else from the world.

        Inside an attempt the world's active attempt id is used. Outside one (a Check or the
        final goal refresh) a fresh observation-only id is minted, never a completed attempt's.
        """
        run_id = self.run_id or self.world.current_run_id or ""
        attempt_id = self.attempt_id or self.world.active_attempt_id
        if self.attempt_id and self.world.active_attempt_id is None:
            attempt_id = ""  # a bound id may belong to a finished attempt
        if not attempt_id:
            self._observe_count += 1
            attempt_id = f"{run_id}:observe:{self._observe_count}"
        await self.observe(run_id, attempt_id)

    async def observe(self, run_id: str, attempt_id: str) -> SceneSnapshot:
        import hashlib
        from datetime import datetime, timezone

        from liminal.perception.evidence import EvidenceRef, FactObservation

        if not run_id or not attempt_id:
            raise ValueError("ReplayObserver needs a run_id and attempt_id (see bind/make_refresh)")
        snapshot = self.scene.capture()
        image = next(item for item in reversed(snapshot.images) if item.event == "schematic_current_scene")
        self._frame_seq += 1
        observed_mono = snapshot.captured_mono_s
        evidence = EvidenceRef(
            evidence_id=image.frame_id,
            artifact_uri=f"replay://{image.frame_id}",
            sha256=hashlib.sha256(image.data).hexdigest(),
            camera_id=image.camera,
            capture_utc=datetime.fromtimestamp(image.captured_at, timezone.utc).isoformat(),
            frame_seq=self._frame_seq,
            stream_epoch=self._epoch,
            capture_mono_s=observed_mono,
        )
        for predicate, value in snapshot.facts.items():
            self.world.observe(FactObservation(
                predicate=predicate,
                value=value,
                evidence=(evidence,),
                observer_id=self.OBSERVER_ID,
                observer_version=self.VERSION,
                observed_utc=evidence.capture_utc,
                observed_mono_s=observed_mono,
                run_id=run_id,
                attempt_id=attempt_id,
                object_bindings=snapshot.bindings[predicate] if self.bound else (),
                reason=f"scene revision {snapshot.revision}: {value.value}",
            ), self._writer)
        self.last_snapshot = snapshot
        self.refresh_count += 1
        return snapshot


class ReplayPrimitives:
    """PrimitiveBackend that also tells the replay scene when a support arm lets go.

    ``release`` and ``home`` drop the position hold, so the states that arm kept up (the lid)
    fall in the scene. The arm's motion and safety path are the real primitive's.
    """

    LETS_GO = frozenset({"release", "home"})

    def __init__(self, scene: ReplayScene, inner: Any):
        self.scene = scene
        self.inner = inner

    async def execute(self, skill: Skill, ctx: Any) -> SkillResult:
        result = await self.inner.execute(skill, ctx)
        if skill.name in self.LETS_GO and result.status.ok:
            for arm in ctx.members(skill.arm):
                self.scene.support_released(arm)
        return result


def attach_replay_primitives(ctx: Any, scene: ReplayScene) -> ReplayPrimitives:
    """Wrap ``ctx.skills.primitives`` so releasing a support arm is visible to the scene."""
    wrapped = ReplayPrimitives(scene, ctx.skills.primitives)
    ctx.skills.primitives = wrapped
    return wrapped


def planning_frames(snapshot: SceneSnapshot) -> tuple[Any, ...]:
    """Planner ObservationFrames: the live frame is current; reference frames are historical."""
    from liminal.orchestration.plan_validation import ObservationFrame

    return tuple(
        ObservationFrame(
            image.frame_id, image.camera, image.captured_at,
            "current" if image.event == "schematic_current_scene" else "historical",
            image.media_type, image.data,
            snapshot.revision if image.event == "schematic_current_scene" else None,
        )
        for image in sorted(snapshot.images, key=lambda item: item.event == "schematic_current_scene")
    )


def _required_string(mapping: Mapping[str, Any], key: str, where: str) -> str:
    return _nonempty_string(mapping.get(key), f"{where}.{key}")


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _string_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{where} must be a list of non-empty strings")
    return value


def _render_png(scenario: ReplayScenario, locations: Mapping[str, str], *, hidden: Collection[str] = (),
                marked: str | None = None, states: Mapping[str, bool] | None = None) -> bytes:
    """Draw a tiny schematic work area using only the standard library."""
    width, height = 320, 180
    background = (37, 44, 54)
    pixels = [[background for _ in range(width)] for _ in range(height)]
    _rect(pixels, 18, 16, 284, 148, (190, 194, 185))
    # One container per destination object. A single destination keeps the original geometry; several
    # get their own distinct region and frame colour (left_bin left, right_bin right).
    movers = {b["object_id"] for b in scenario.obligations.values()}
    destinations = sorted(
        ({b["destination_id"] for b in scenario.obligations.values()}
         | {o for o, label in scenario.objects.items() if o not in movers and "block" not in label.lower()}
         | {loc for loc in locations.values() if loc in scenario.objects}) - movers)
    containers: dict[str, tuple[int, int, int]] = {}  # destination -> (x, y, size)
    if len(destinations) <= 1:
        for name in destinations:
            containers[name] = (222, 56, 58)
    else:
        spots = [(226, 22), (200, 86), (252, 86)]
        free = [i for i in range(len(spots))]
        for name, preferred in (("left_bin", 1), ("right_bin", 2)):
            if name in destinations and preferred in free:
                containers[name] = (*spots[preferred], 52)
                free.remove(preferred)
        for name in destinations:
            if name not in containers:
                containers[name] = (*spots[free.pop(0) if free else 0], 52)
    frames = [(77, 88, 98), (150, 110, 60), (60, 120, 150), (130, 80, 150)]
    for index, name in enumerate(destinations):
        x, y, size = containers[name]
        _rect(pixels, x, y, size, size, frames[index % len(frames)] if len(destinations) > 1 else (77, 88, 98))
        _rect(pixels, x + 5, y + 5, size - 10, size - 10, (226, 226, 215))
    if not destinations:
        _rect(pixels, 222, 56, 58, 58, (77, 88, 98))
        _rect(pixels, 227, 61, 48, 48, (226, 226, 215))

    for index, key in enumerate(sorted(states or ())):  # one indicator per observed state: green = true
        _rect(pixels, 22 + 14 * index, 20, 10, 10, (60, 170, 80) if states[key] else (150, 60, 60))

    table_slots = [(48, 54), (103, 54), (158, 54), (48, 112), (103, 112), (158, 112)]
    box_slots = [(236, 68), (253, 68), (236, 87), (253, 87)]
    box_index: dict[str, int] = {}
    for name in sorted({loc for loc in locations.values() if loc in scenario.objects and loc != "table"}):
        for index, object_id in enumerate(sorted(o for o, loc in locations.items() if loc == name)):
            box_index[object_id] = index
    table_index = 0
    for object_id in sorted(locations):
        location = locations[object_id]
        if location == "table":
            x, y = table_slots[table_index % len(table_slots)]
            table_index += 1
        elif location in scenario.objects:
            if len(destinations) > 1 and location in containers:
                cx, cy, _ = containers[location]
                k = box_index[object_id] % 4
                x, y = cx + 5 + 20 * (k % 2) - 2, cy + 5 + 20 * (k // 2) - 2
            else:
                x, y = box_slots[box_index[object_id] % len(box_slots)]
        else:
            continue
        label = scenario.objects.get(object_id, object_id).lower()
        color = ((220, 55, 55) if "red" in label else (52, 93, 218) if "blue" in label
                 else (60, 170, 80) if "green" in label else (226, 177, 49))
        if object_id in hidden:
            _rect(pixels, x - 4, y - 4, 27, 27, (96, 96, 96))  # occluder: the object is not visible
            continue
        if object_id == marked:
            _rect(pixels, x - 3, y - 3, 25, 25, (255, 255, 255))  # reference frame: the indicated target
        _rect(pixels, x, y, 19, 19, color)
        _rect(pixels, x + 3, y + 3, 13, 13, tuple(min(255, channel + 25) for channel in color))
    raw = b"".join(b"\x00" + bytes(channel for pixel in row for channel in pixel) for row in pixels)
    return b"\x89PNG\r\n\x1a\n" + \
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + \
        _png_chunk(b"IDAT", zlib.compress(raw)) + _png_chunk(b"IEND", b"")


def _rect(pixels: list[list[tuple[int, int, int]]], x: int, y: int, width: int, height: int,
          color: tuple[int, int, int]) -> None:
    for row in range(max(0, y), min(len(pixels), y + height)):
        for column in range(max(0, x), min(len(pixels[0]), x + width)):
            pixels[row][column] = color


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
