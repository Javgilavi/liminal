"""Interruptible, evidence-verified episode runner (WS3, V2/V3).

One loop ties the pieces together: observe -> reconcile the ledger -> plan one bounded step
-> dispatch it inside an AttemptScope -> verify from fresh evidence -> remember.

Authority never moves to the model: the planner only proposes (and is validated), the tree
executes, and an obligation is complete only when a fresh observer record says so. Backend and
skill success claims are never written to the ledger. A restart reconciles interrupted attempts
from a new observation and never resends them.

Trace events (all through ``world.log``): episode_start, episode_observation, episode_ledger,
episode_context, episode_proposal, episode_plan_dropped, episode_dispatch, episode_command_outcome,
episode_verification, episode_memory, episode_keyframes, episode_experience, episode_stop.

Memory (see ``liminal.memory``): ``keyframes`` picks the historical frames the planner sees ("event" =
verified-ledger moments, "nominated" = MemER-style planner nominations clustered over time, "none" =
recent frames and the goal reference only). Every planning context also carries a progress record and
text summary built only from verified ledger facts. ``experience_db``/``experience_mode`` add Harness-VLA
style cross-run procedures and failure records as advisory context; they never mark progress.
"""

from __future__ import annotations

import inspect
import sqlite3
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from liminal.memory import ExperienceStore, KeyframeMemory, MemoryStore, build_progress
from liminal.memory.experience import EXPERIENCE_MODES, classify_failure, collect_run_experience, hit_ids
from liminal.memory.inspection import CONFIG_KIND, EXPERIENCE_KIND
from liminal.memory.keyframes import KEYFRAME_MODES, plain
from liminal.orchestration.attempt import AttemptScope
from liminal.orchestration.context import ExecutionContext
from liminal.orchestration.executor import run_task
from liminal.orchestration.plan_validation import (
    EvidenceRecord,
    ObservationFrame,
    PlanBudget,
    PlanningContext,
    PlanValidationError,
    ValidatedPlan,
)
from liminal.orchestration.planner import ConstrainedPlanner, TextModel
from liminal.perception.evidence import Truth, normalize_bindings
from liminal.perception.verifier import Verifier
from liminal.safety.supervisor import EmergencyStop

# Observer refresh: write fresh evidence for (run_id, attempt_id) into the world, return a snapshot
# with .revision, .bindings {predicate: {name: id}} and .images (frame_id, camera, captured_at,
# data, media_type, event). ReplayObserver.observe has this shape.
Observe = Callable[[str, str], Awaitable[Any] | Any]

REFERENCE_PREFIX = "goal_reference"
MAX_BRACKET_FRAMES = 4  # keyframes kept as selected memory (pre/post of verified attempts in "event" mode)


class _Scenario(Protocol):
    name: str
    goal: Any
    objects: Any
    registry: Any


class EpisodeBudgetExhausted(RuntimeError):
    def __init__(self, budget: str):
        super().__init__(f"budget exhausted: {budget}")
        self.budget = budget


@dataclass(frozen=True)
class ObligationStatus:
    obligation_id: str
    status: str  # "verified" | "pending" | "unknown"
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AttemptSummary:
    attempt_id: str
    status: str
    obligation_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    skill: str = ""
    arm: str = ""
    prompt: str = ""


@dataclass
class EpisodeReport:
    run_id: str
    goal_id: str
    stop_reason: str
    completed: bool
    obligations: tuple[ObligationStatus, ...]
    attempts: tuple[AttemptSummary, ...]
    verified_effects: tuple[str, ...]
    attempts_used: float = 0
    model_calls_used: float = 0
    session_verified_attempts: int = 0
    resumed: bool = False
    mode: str = "replay"
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        label = "REPLAY/MOCK, not hardware" if self.mode != "hardware" else "hardware"
        lines = [f"run {self.run_id} [{label}]: {self.stop_reason} (completed={self.completed})"]
        lines += [f"  {o.obligation_id}: {o.status} {list(o.evidence_ids)}" for o in self.obligations]
        lines += [f"  {a.attempt_id}: {a.status} {a.skill}/{a.arm} {a.prompt}".rstrip() for a in self.attempts]
        return "\n".join(lines)


class _BudgetedModel:
    """Counts every provider call against the run's durable model-call budget."""

    def __init__(self, inner: TextModel, memory: MemoryStore, run_id: str, limit: int):
        self._inner, self._memory, self._run_id, self._limit = inner, memory, run_id, limit
        if getattr(inner, "complete_context", None) is not None:
            self.complete_context = self._complete_context

    def _charge(self) -> None:
        try:
            self._memory.consume_budget(self._run_id, "model_calls", 1, limit=self._limit)
        except ValueError as exc:
            if "budget exhausted" in str(exc):
                raise EpisodeBudgetExhausted("model_calls") from exc
            raise

    def complete(self, system: str, user: str, images: Sequence[bytes] = ()) -> str:
        self._charge()
        return self._inner.complete(system, user, images)

    def _complete_context(self, system: str, user: str, frames: Sequence[Any]) -> str:
        self._charge()
        return self._inner.complete_context(system, user, frames)  # type: ignore[attr-defined]


def make_model(kind: str, scenario: Any = None, scene: Any = None) -> TextModel:
    """`replay` needs no key or network; `claude` reads .env (never required for replay)."""
    if kind == "replay":
        from liminal.orchestration.replay import ReplayModel

        return ReplayModel(scenario, scene)
    if kind == "claude":
        import os

        from liminal import settings
        from liminal.orchestration.planner import ClaudeModel

        settings.load_env_file()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example); use --model replay offline")
        return ClaudeModel(settings.claude_model(),
                           max_tokens=settings.positive_setting("LIMINAL_MAX_OUTPUT_TOKENS", 1500),
                           timeout_s=float(settings.positive_setting("LIMINAL_MODEL_TIMEOUT_S", 30)))
    raise ValueError(f"unknown model {kind!r} (replay | claude)")


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _skill_dicts(node: Any) -> list[dict[str, Any]]:
    """Skill nodes of a rendered task, from its raw (already validated) dict form."""
    found: list[dict[str, Any]] = []
    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if isinstance(item.get("skill"), dict):
                found.append(item["skill"])
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return found


class _Episode:
    def __init__(self, scenario: _Scenario, ctx: ExecutionContext, model: TextModel, memory: MemoryStore,
                 observe: Observe, observer_ids: Sequence[str], run_id: str, *, max_attempts: int,
                 max_model_calls: int, interrupt_after: int | None, plan_deadline_s: float | None,
                 verifier: Verifier | None, max_context_frames: int, mode: str, keyframes: str = "event",
                 max_keyframes: int = MAX_BRACKET_FRAMES, keyframe_gap: int = 1,
                 experience: ExperienceStore | None = None, experience_mode: str = "off"):
        self.scenario, self.ctx, self.memory, self.observe = scenario, ctx, memory, observe
        self.world = ctx.world
        self.observer_ids = tuple(observer_ids)
        self.run_id, self.mode = run_id, mode
        self.goal = scenario.goal
        self.interrupt_after, self.plan_deadline_s = interrupt_after, plan_deadline_s
        self.verifier = verifier or Verifier()
        self.max_context_frames = max_context_frames
        self.max_attempts = self._limit("attempts", max_attempts)
        self.max_model_calls = self._limit("model_calls", max_model_calls)
        self.planner = ConstrainedPlanner(_BudgetedModel(model, memory, run_id, self.max_model_calls))
        self.session_verified = 0
        self.resumed = False
        self.notes: list[str] = []
        self.snapshot: Any = None
        self.arm_hints: dict[str, str] = {}  # operator-accepted effect -> arm; enforced by the validator
        self.kf = KeyframeMemory(memory, run_id, keyframes, max_keyframes=max_keyframes, gap=keyframe_gap)
        self.experience, self.experience_mode = experience, experience_mode
        stored = [f.metadata.get("seq", 0) for f in (*memory.recent_frames(run_id), *memory.selected_frames(run_id))]
        self._frame_seq = max((s for s in stored if isinstance(s, int)), default=0)

    # -- helpers ----------------------------------------------------------------------------

    def _limit(self, name: str, requested: int) -> int:
        """Budgets are durable per run and can only shrink across a resume."""
        _, existing = self.memory.budget(self.run_id, name)
        return int(requested if existing is None else min(requested, existing))

    def _remaining(self, name: str, limit: int) -> int:
        consumed, _ = self.memory.budget(self.run_id, name)
        return int(limit - consumed)

    def log(self, kind: str, **data: Any) -> None:
        self.world.log(f"episode_{kind}", run_id=self.run_id, mode=self.mode, **data)

    async def _observe(self, tag: str = "obs") -> Any:
        snapshot = await _maybe_await(self.observe(self.run_id, f"{tag}-{uuid.uuid4().hex[:8]}"))
        self.snapshot = snapshot
        self._remember_snapshot(snapshot)
        self.log("observation", revision=snapshot.revision,
                 facts={p: self._truth(p)[0].value for p in self.goal.obligations},
                 frames=[i.frame_id for i in snapshot.images])
        return snapshot

    @staticmethod
    def _is_reference(image: Any) -> bool:
        return str(image.event).startswith(REFERENCE_PREFIX)

    def _add_frame(self, image: Any, revision: int | None, role: str) -> None:
        try:
            self.memory.get_frame(self.run_id, image.frame_id)
            return
        except KeyError:
            pass
        self._frame_seq += 1  # the run's own time axis, which keyframe clustering measures gaps on
        self.memory.add_frame(self.run_id, image.frame_id, image.data, captured_at=image.captured_at,
                              camera=image.camera, event=image.event,
                              metadata={"media_type": image.media_type, "revision": revision, "role": role,
                                        "seq": self._frame_seq})

    def _remember_snapshot(self, snapshot: Any) -> None:
        refs = [i for i in snapshot.images if self._is_reference(i)]
        for image in snapshot.images:
            self._add_frame(image, None if self._is_reference(image) else snapshot.revision,
                            "reference" if self._is_reference(image) else "observation")
        if refs:
            # Selected memory survives pruning, so the goal reference outlives the recent window.
            self.kf.apply(add_references=[i.frame_id for i in refs])

    def _truth(self, predicate: str) -> tuple[Truth, tuple[str, ...]]:
        bindings = (self.snapshot.bindings.get(predicate) if self.snapshot is not None else None)
        truth = self.world.evaluate(predicate, bindings=bindings)
        records = self.world.evidence_records(predicate, bindings)
        return truth, (records[-1].evidence_ids if records else ())

    def _ensure_evidence(self, evidence_ids: Sequence[str], predicate: str, truth: Truth, kind: str = "frame") -> None:
        for evidence_id in evidence_ids:
            try:
                self.memory.get_evidence(self.run_id, evidence_id)
            except KeyError:
                self.memory.record_evidence(self.run_id, evidence_id, kind, {
                    "predicate": predicate, "value": truth.value,
                    "revision": getattr(self.snapshot, "revision", None)})

    # -- ledger -----------------------------------------------------------------------------

    def _reconcile_attempts(self) -> bool:
        """Resolve interrupted/unknown attempts from the fresh observation. False if one stays open."""
        resolved = True
        for attempt_id in self.memory.reconcile(self.run_id).unresolved_attempt_ids:
            attempt = self.memory.get_attempt(self.run_id, attempt_id)
            # Only crash leftovers are reconciled. An attempt we already judged ('unknown'/'failed') is
            # never retro-credited from a later observation, which may stem from a different attempt.
            if attempt.status != "interrupted":
                continue
            facts = {o: self._truth(o) for o in attempt.obligation_ids}
            for obligation_id, (truth, evidence) in facts.items():
                if truth is not Truth.UNKNOWN:
                    self._ensure_evidence(evidence, obligation_id, truth)
            if any(t is Truth.UNKNOWN for t, _ in facts.values()) or not facts:
                resolved = False
                self.log("ledger", action="attempt_unresolved", attempt_id=attempt_id, status=attempt.status)
                continue
            evidence = tuple(dict.fromkeys(e for _, ev in facts.values() for e in ev))
            if all(t is Truth.TRUE for t, _ in facts.values()):
                only = next(iter(facts)) if len(facts) == 1 else None
                if only and self.memory.get_obligation(self.run_id, only).status == "completed":
                    only = None  # already credited by a verified effect; never duplicate it
                self.memory.reconcile_attempt(
                    self.run_id, attempt_id, "verified", evidence_ids=evidence,
                    verified_obligation_ids=tuple(facts), effect_key=self._effect_key(only) if only else None,
                    effect_payload={"predicate": only, "reconciled_from": attempt.status} if only else None)
                status = "verified"
            else:
                self.memory.reconcile_attempt(self.run_id, attempt_id, "failed", evidence_ids=evidence)
                status = "failed"
            self.log("ledger", action="attempt_reconciled", attempt_id=attempt_id, was=attempt.status, now=status,
                     evidence_ids=evidence, resent=False)
        return resolved

    def _effect_key(self, predicate: str) -> str:
        """Unique per completion: an obligation undone in the world and redone is a new effect."""
        done = sum(1 for e in self.memory.verified_effects(self.run_id) if e.payload.get("predicate") == predicate)
        return f"{predicate}#{done + 1}"

    def _reconcile_obligations(self) -> None:
        for predicate in self.goal.obligations:
            truth, evidence = self._truth(predicate)
            status = self.memory.get_obligation(self.run_id, predicate).status
            if truth is Truth.UNKNOWN or not evidence:
                continue  # absence of evidence never completes, and never reopens, an obligation
            self._ensure_evidence(evidence, predicate, truth)
            if truth is Truth.TRUE and status != "completed":
                self.memory.complete_obligation(self.run_id, predicate, evidence)
                self.log("ledger", action="obligation_completed", obligation=predicate, evidence_ids=evidence)
            elif truth is Truth.FALSE and status == "completed":
                self.memory.set_obligation_status(self.run_id, predicate, "pending", evidence_ids=evidence)
                self.log("ledger", action="obligation_reopened", obligation=predicate, evidence_ids=evidence)

    def _pending(self) -> list[str]:
        done = {o.obligation_id for o in self.memory.obligations(self.run_id, "completed")}
        return [o for o in self.goal.obligations if o not in done]

    # -- planning ---------------------------------------------------------------------------

    def _planning_context(self, remaining_model_calls: int) -> PlanningContext:
        snap = self.snapshot
        current = [i for i in snap.images if not self._is_reference(i)]
        frames = [ObservationFrame(i.frame_id, i.camera, i.captured_at, "current", i.media_type, i.data, snap.revision)
                  for i in current]
        cap = self.max_context_frames
        current_ids = {f.frame_id for f in frames}
        history = self.kf.context_frames(max(0, cap - len(frames)), exclude=current_ids)
        for record in history:
            frames.append(ObservationFrame(
                record.frame_id, record.camera, record.captured_at, "historical",
                str(record.metadata.get("media_type", "image/png")), record.data))
        selection = self.memory.persist_context(self.run_id, [f.frame_id for f in frames],
                                                reason="episode planning context")
        evidence = []
        for predicate in self.goal.obligations:
            truth, ids = self._truth(predicate)
            binding = self.snapshot.bindings.get(predicate, {})
            evidence.append(EvidenceRecord(f"fact:{predicate}:{snap.revision}", f"observed_{truth.value}",
                                           tuple(binding.values()), predicate, "current", snap.revision))
        pending = self._pending()
        completed = [o for o in self.goal.obligations if o not in pending]
        progress = {
            "completed_effects": completed,
            "pending_effects": [{"predicate": p, **snap.bindings.get(p, {})} for p in pending],
            "attempts_used": self.memory.budget(self.run_id, "attempts")[0],
            "attempts_remaining": self._remaining("attempts", self.max_attempts),
        }
        if self.arm_hints:
            progress["operator_arms"] = {e: a for e, a in self.arm_hints.items() if e in pending}
        budget = PlanBudget(max_model_calls=max(1, remaining_model_calls), max_context_frames=cap)
        memory = self._memory_payload(history)
        experience = self._retrieve_experience(pending)
        self.log("context", frames=[{"frame_id": f.frame_id, "role": f.temporal_role} for f in frames],
                 selected_memory=[f.frame_id for f in selection.frames], evidence=[e.evidence_id for e in evidence],
                 progress=progress, revision=snap.revision, keyframes=self.kf.mode,
                 summary=memory["summary"])
        return PlanningContext(self.run_id, snap.revision, self.goal, tuple(frames), tuple(evidence),
                               dict(self.scenario.objects), self.scenario.registry, budget, progress,
                               memory=memory, experience=experience,
                               nominate_keyframes=self.kf.mode == "nominated")

    def _memory_payload(self, history: Sequence[Any]) -> dict[str, Any]:
        """Planner-facing memory: ledger-only progress record, its summary, and why each old frame is shown."""
        record = build_progress(self.memory, self.run_id)
        summary = record.pop("summary")
        kept = {n.note_id: plain(n.payload) for n in self.memory.notes(self.run_id, "keyframe_selected")}
        frames = []
        for frame in history:
            entry = kept.get(frame.frame_id)
            frames.append({"frame_id": frame.frame_id,
                           "kind": "goal_reference" if entry and entry["source"] == "reference"
                           else "keyframe" if entry else "recent",
                           "reason": entry["reason"] if entry else "recent observation window"})
        return {"progress_record": record, "summary": summary, "historical_frames": frames,
                "note": "Built from verified ledger facts only; historical frames never certify the present scene."}

    def _retrieve_experience(self, pending: Sequence[str]) -> dict[str, Any]:
        if self.experience is None or self.experience_mode == "off":
            return {}
        roles = {v for p in pending for v in (self.snapshot.bindings.get(p) or {}).values()}
        roles = roles or set(self.scenario.objects)
        registry = self.scenario.registry
        hits = self.experience.retrieve(registry_version=str(registry.version), arms=registry.available_arms,
                                        roles=roles, exclude_run_id=self.run_id)
        ids = hit_ids(hits)
        self.log("experience", action="retrieved", experience_mode=self.experience_mode, frozen=self.experience.frozen,
                 corpus_hash=hits["corpus_hash"], hit_ids=ids, roles=sorted(roles))
        self.memory.put_note(self.run_id, EXPERIENCE_KIND, "latest", {"hit_ids": ids, **hits, "roles": sorted(roles)})
        return hits if ids else {}

    # -- dispatch ---------------------------------------------------------------------------

    async def _dispatch(self, plan: ValidatedPlan) -> bool:
        """Run one validated step as a single attempt. Returns True when it is verified."""
        skills = _skill_dicts(plan.task.raw.get("tree"))
        step = skills[-1] if skills else {}
        effect = plan.task.goal or ""
        attempt_no = len(self.memory.attempts(self.run_id)) + 1
        attempt_id = f"att-{attempt_no:03d}"
        bindings = normalize_bindings(self.snapshot.bindings.get(effect))
        summary = {"skill": step.get("name"), "arm": step.get("arm"), "prompt": step.get("prompt"),
                   "object_id": step.get("target"), "effect": effect,
                   "destination_id": (step.get("params") or {}).get("destination_id")}
        try:
            self.memory.consume_budget(self.run_id, "attempts", 1, limit=self.max_attempts)
        except ValueError as exc:
            raise EpisodeBudgetExhausted("attempts") from exc
        self.memory.start_attempt(self.run_id, attempt_id, [effect], {**summary, "revision": plan.base_revision})
        self.memory.record_event(self.run_id, f"{attempt_id}:proposal", "proposal",
                                 {**summary, "evidence_refs": list(plan.used_evidence),
                                  "sketch": list(plan.dependency_sketch)}, attempt_id=attempt_id)
        pre_frames = [i for i in self.snapshot.images if not self._is_reference(i)]
        self.log("dispatch", attempt_id=attempt_id, **summary)
        captured: list[Any] = []

        async def refresh() -> None:
            captured.append(await _maybe_await(self.observe(self.run_id, attempt_id)))

        self.world.begin_attempt(attempt_id)
        scope = AttemptScope(self.run_id, attempt_id, effect, bindings, self.observer_ids, refresh, self.verifier)
        self.ctx.verification = scope
        try:
            report = await run_task(plan.task, self.ctx)
        finally:
            # A crash here leaves the attempt 'started' in SQLite; reopening marks it interrupted.
            self.ctx.verification = None
            self.world.end_attempt()
        self.log("command_outcome", attempt_id=attempt_id, status=report.status.value,
                 backend_reported_goal=report.goal_verified, error=report.error)
        result = scope.result
        decision = result.decision if result is not None else None
        evidence = tuple(result.evidence_ids) if result is not None else ()
        if result is not None:
            self.log("verification", attempt_id=attempt_id, decision=result.decision.value,
                     evidence_ids=evidence, observer=result.observer_id, reason=result.reason)
        if captured:
            self.snapshot = captured[-1]
        verified = decision is Truth.TRUE and report.goal_verified and bool(evidence)
        status = "verified" if verified else ("failed" if decision in (None, Truth.FALSE) else "unknown")
        if evidence:
            self._ensure_evidence(evidence, effect, decision or Truth.UNKNOWN, kind="verification")
        if verified:
            self.memory.finish_attempt(
                self.run_id, attempt_id, "verified", evidence_ids=evidence, verified_obligation_ids=[effect],
                effect_key=self._effect_key(effect),
                effect_payload={"predicate": effect, "bindings": dict(bindings), **summary})
            post = [i for i in self.snapshot.images if not self._is_reference(i)]
            for image in post:
                self._add_frame(image, self.snapshot.revision, "post_attempt")
            self._note_event_frames([i.frame_id for i in pre_frames[-1:]], "pre_attempt", attempt_id, effect)
            self._note_event_frames([i.frame_id for i in post[-1:]], "post_verified", attempt_id, effect)
            self.session_verified += 1
        else:
            self.memory.finish_attempt(self.run_id, attempt_id, status, evidence_ids=evidence)
            if self.kf.mode == "event":
                post = [i for i in self.snapshot.images if not self._is_reference(i)]
                for image in post:
                    self._add_frame(image, self.snapshot.revision, "post_attempt")
                self._note_event_frames([i.frame_id for i in post[-1:]], "failure", attempt_id, effect)
        if not verified:
            kind = classify_failure(decision=decision.value if decision is not None else None, error=report.error)
            self.memory.record_event(
                self.run_id, f"{attempt_id}:outcome", "attempt_outcome",
                {"status": status, "failure_kind": kind, "reason": (result.reason if result is not None
                                                                     else (report.error or report.status.value))[:160],
                 "backend_status": report.status.value}, attempt_id=attempt_id)
        self.log("memory", attempt_id=attempt_id, status=status, verified=verified,
                 obligations={o.obligation_id: o.status for o in self.memory.obligations(self.run_id)},
                 verified_effects=[e.effect_key for e in self.memory.verified_effects(self.run_id)])
        if report.error and "EmergencyStop" in report.error:
            raise EmergencyStop(report.error)
        return verified

    def _note_event_frames(self, frame_ids: list[str], event: str, attempt_id: str, effect: str) -> None:
        recorded = self.kf.note_event(frame_ids, event, f"{attempt_id} {effect}", attempt_id)
        if recorded or self.kf.mode == "event":
            selected = self.kf.apply()
            self.log("keyframes", action="event", event=event, attempt_id=attempt_id, frame_ids=recorded,
                     selected=[{"frame_id": e["frame_id"], "reason": e["reason"]} for e in selected])

    def _absorb_nominations(self, plan: ValidatedPlan) -> None:
        """Planner keyframe nominations: advisory, already limited to frames the planner was shown."""
        if self.kf.mode != "nominated":
            return
        self.kf.record_rejected(plan.rejected_nominations)
        accepted = self.kf.accept_planner_nominations(plan.keyframe_nominations)
        selected = self.kf.apply()
        self.log("keyframes", action="nominations", accepted=accepted,
                 rejected=[dict(r) for r in plan.rejected_nominations],
                 selected=[{"frame_id": e["frame_id"], "reason": e["reason"], "cluster_size": e["cluster_size"]}
                           for e in selected])

    def _record_drop(self, reason: str, plan: ValidatedPlan, step: dict[str, Any]) -> None:
        """Dropped plans are ledger events too, so a stale plan can become a cross-run failure record."""
        params = step.get("params") or {}
        count = sum(1 for e in self.memory.events(self.run_id) if e.kind == "plan_dropped")
        self.memory.record_event(
            self.run_id, f"drop-{count + 1:03d}", "plan_dropped",
            {"reason": reason, "skill": step.get("name"), "arm": step.get("arm"), "object_id": step.get("target"),
             "destination_id": params.get("destination_id"), "effect": plan.task.goal or ""})

    # -- loop -------------------------------------------------------------------------------

    async def run(self) -> str:
        max_cycles = 2 * (self.max_attempts + self.max_model_calls) + 4
        for _ in range(max_cycles):
            await self._observe()
            if not self._reconcile_attempts():
                return "unresolved_attempt"
            self._reconcile_obligations()
            if not self._pending():
                return "completed"
            if self._remaining("attempts", self.max_attempts) < 1:
                return "budget_attempts"
            calls_left = self._remaining("model_calls", self.max_model_calls)
            if calls_left < 1:
                return "budget_model_calls"
            context = self._planning_context(calls_left)
            try:
                plan = await self.planner.plan_async(context, self.plan_deadline_s)
            except EpisodeBudgetExhausted as exc:
                return f"budget_{exc.budget}"
            except PlanValidationError as exc:
                self.log("plan_dropped", reason="invalid", error=str(exc)[:300])
                return "planner_failed"
            step = (_skill_dicts(plan.task.raw.get("tree")) or [{}])[-1]
            self._absorb_nominations(plan)
            self.log("proposal", base_revision=plan.base_revision, steps=list(plan.step_ids),
                     skill=step.get("name"), arm=step.get("arm"), prompt=step.get("prompt"),
                     effect=plan.task.goal, evidence_refs=list(plan.used_evidence),
                     model_calls=self.planner.last_model_call_count)
            # The scene may have moved while the model was thinking: refuse a stale plan.
            live = await self._observe("recheck")
            if plan.base_revision != live.revision or plan.run_id != self.run_id:
                self.log("plan_dropped", reason="stale_revision", plan_revision=plan.base_revision,
                         live_revision=live.revision)
                self._record_drop("stale_revision", plan, step)
                continue
            if plan.task.goal not in self._pending() or self._truth(plan.task.goal or "")[0] is Truth.TRUE:
                self.log("plan_dropped", reason="effect_not_pending", effect=plan.task.goal)
                self._record_drop("effect_not_pending", plan, step)
                continue
            try:
                await self._dispatch(plan)
            except EpisodeBudgetExhausted as exc:
                return f"budget_{exc.budget}"
            except EmergencyStop:
                return "safety_estop"
            if self.interrupt_after is not None and self.session_verified >= self.interrupt_after:
                return "interrupted"
        return "max_cycles"

    def finish_experience(self, stop_reason: str) -> None:
        """Run end: distil the verified trace and failure records into the cross-run store (if allowed)."""
        if self.experience is None or self.experience_mode == "off" or stop_reason == "interrupted":
            return  # an interrupted run is paused, not finished: its resume writes the record
        if not self.experience.writable:
            self.log("experience", action="not_written", experience_mode=self.experience_mode,
                     frozen=self.experience.frozen,
                     reason="frozen corpus" if self.experience.frozen else "read-only mode")
            return
        steps, failures, completed = collect_run_experience(self.memory, self.run_id)
        registry = self.scenario.registry
        self.experience.record_run(
            self.run_id, scenario=self.scenario.name, registry_version=str(registry.version),
            arms=registry.available_arms, goal_text=self.goal.text, completed=completed and stop_reason == "completed",
            steps=steps, failures=failures)
        self.log("experience", action="written", experience_mode=self.experience_mode, steps=len(steps),
                 failures=[f["kind"] for f in failures], corpus_hash=self.experience.corpus_hash())

    def report(self, stop_reason: str) -> EpisodeReport:
        obligations = []
        for o in self.memory.obligations(self.run_id):
            status = {"completed": "verified"}.get(o.status, o.status)
            obligations.append(ObligationStatus(o.obligation_id, status, o.evidence_ids))
        attempts = tuple(AttemptSummary(a.attempt_id, a.status, a.obligation_ids, a.evidence_ids,
                                        str(a.payload.get("skill") or ""), str(a.payload.get("arm") or ""),
                                        str(a.payload.get("prompt") or ""))
                         for a in self.memory.attempts(self.run_id))
        return EpisodeReport(
            self.run_id, self.goal.goal_id, stop_reason, stop_reason == "completed", tuple(obligations), attempts,
            tuple(str(e.payload.get("predicate") or e.effect_key) for e in self.memory.verified_effects(self.run_id)),
            self.memory.budget(self.run_id, "attempts")[0], self.memory.budget(self.run_id, "model_calls")[0],
            self.session_verified, self.resumed, self.mode, self.notes)


async def run_episode(
    scenario: _Scenario,
    ctx: ExecutionContext,
    model: TextModel,
    memory: MemoryStore | str | Path,
    observe: Observe,
    *,
    observer_ids: Sequence[str],
    run_id: str | None = None,
    max_attempts: int = 6,
    max_model_calls: int = 20,
    interrupt_after: int | None = None,
    plan_deadline_s: float | None = None,
    verifier: Verifier | None = None,
    max_context_frames: int = 6,
    mode: str = "replay",
    keyframes: str = "event",
    max_keyframes: int = MAX_BRACKET_FRAMES,
    keyframe_gap: int = 1,
    experience_db: str | Path | ExperienceStore | None = None,
    experience_mode: str = "off",
    experience_frozen: bool = False,
    arm_hints: Mapping[str, str] | None = None,
) -> EpisodeReport:
    """Run (or resume, when ``run_id`` already exists in the db) one goal to verified completion.

    Memory options: ``keyframes`` "event" | "nominated" | "none" (see module docstring), at most
    ``max_keyframes`` of them, ``keyframe_gap`` frames apart before nominations cluster. ``experience_db``
    (a path or an ExperienceStore) with ``experience_mode`` "off" | "read" | "readwrite" enables advisory
    cross-run experience; ``experience_frozen`` makes it read-only whatever the mode.

    ``memory`` may be an open store or a db path; a path is opened with recover_open_attempts=True and
    closed here. ``observe(run_id, attempt_id)`` must write observer evidence into ``ctx.world`` and
    return a snapshot. ``mode`` labels the report/trace so mock results are never read as hardware.
    """
    if keyframes not in KEYFRAME_MODES:
        raise ValueError(f"unknown keyframes mode {keyframes!r} (expected one of {KEYFRAME_MODES})")
    if experience_mode not in EXPERIENCE_MODES:
        raise ValueError(f"unknown experience_mode {experience_mode!r} (expected one of {EXPERIENCE_MODES})")
    if experience_mode != "off" and experience_db is None:
        raise ValueError("experience_mode needs an experience_db path")
    owns_store = not isinstance(memory, MemoryStore)
    store = MemoryStore(memory, recover_open_attempts=True) if owns_store else memory
    experience: ExperienceStore | None = None
    owns_experience = False
    if experience_mode != "off":
        if isinstance(experience_db, ExperienceStore):
            experience = experience_db
        else:
            writable = experience_mode == "readwrite" and not experience_frozen
            experience = ExperienceStore(experience_db, writable=writable, frozen=experience_frozen)  # type: ignore[arg-type]
            owns_experience = True
    try:
        run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        goal = scenario.goal
        resumed = _run_exists(store, run_id)
        store.create_run(run_id, {"goal_id": goal.goal_id, "text": goal.text, "obligations": list(goal.obligations)},
                         {"scenario": scenario.name, "registry_version": scenario.registry.version})
        store.set_run_status(run_id, "active")
        episode = _Episode(scenario, ctx, model, store, observe, observer_ids, run_id,
                           max_attempts=max_attempts, max_model_calls=max_model_calls,
                           interrupt_after=interrupt_after, plan_deadline_s=plan_deadline_s, verifier=verifier,
                           max_context_frames=max_context_frames, mode=mode, keyframes=keyframes,
                           max_keyframes=max_keyframes, keyframe_gap=keyframe_gap, experience=experience,
                           experience_mode=experience_mode)
        episode.resumed = resumed
        episode.arm_hints = dict(arm_hints or {})
        store.put_note(run_id, CONFIG_KIND, "config", {
            "keyframes": keyframes, "max_keyframes": max_keyframes, "keyframe_gap": keyframe_gap,
            "max_context_frames": max_context_frames, "experience_mode": experience_mode,
            "experience_frozen": bool(experience_frozen or (experience is not None and experience.frozen)),
            "experience_db": str(experience.path) if experience is not None else None})
        ctx.world.begin_run(run_id)
        for obligation in goal.obligations:
            try:
                store.get_obligation(run_id, obligation)
            except KeyError:
                store.add_obligation(run_id, obligation, {"predicate": obligation})
        episode.log("start", goal_id=goal.goal_id, goal=goal.text, obligations=list(goal.obligations),
                    resumed=resumed, max_attempts=episode.max_attempts, max_model_calls=episode.max_model_calls,
                    interrupt_after=interrupt_after)
        try:
            reason = await episode.run()
        except (sqlite3.Error, KeyError, ValueError) as exc:
            store.set_run_status(run_id, "failed")
            episode.log("stop", reason=f"error: {type(exc).__name__}: {exc}")
            raise
        store.set_run_status(run_id, "completed" if reason == "completed" else "paused")
        episode.finish_experience(reason)
        report = episode.report(reason)
        episode.log("stop", reason=reason, completed=report.completed,
                    obligations={o.obligation_id: o.status for o in report.obligations},
                    verified_effects=list(report.verified_effects))
        return report
    finally:
        if owns_experience and experience is not None:
            experience.close()
        if owns_store:
            store.close()


async def run_authored(
    scenario: _Scenario,
    ctx: ExecutionContext,
    task: Any,
    observer: Any,
    *,
    run_id: str | None = None,
    verifier: Verifier | None = None,
    mode: str = "replay",
) -> EpisodeReport:
    """V1: run an authored task tree against a replay scenario. No planner, no ledger.

    ``observer`` must be an unbound ReplayObserver (``bound=False``): authored trees read predicates
    without object bindings. Every skill with a ``success`` predicate runs in its own EvidenceAttempts
    attempt, and the obligations in the report come from a final fresh observation, never from the
    backend or the tree's status.
    """
    from liminal.orchestration.attempt import EvidenceAttempts

    if ctx.verification is not None or ctx.attempts is not None:
        raise ValueError("a context carries either one attempt scope or an attempt factory, not both")
    run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
    ctx.world.begin_run(run_id)
    factory = EvidenceAttempts(ctx.world, run_id, observer.refresh, (observer.OBSERVER_ID,), verifier or Verifier())
    ctx.attempts = factory
    ctx.observe = observer.refresh
    ctx.world.log("episode_start", run_id=run_id, mode=mode, task=task.name, goal=scenario.goal.text,
                  obligations=list(scenario.goal.obligations), authored=True)
    try:
        run = await run_task(task, ctx)
        await observer.refresh()  # final fresh evidence for the report
    finally:
        ctx.attempts = None
        ctx.observe = None
    obligations, attempts = [], []
    for predicate in scenario.goal.obligations:
        truth = ctx.world.evaluate(predicate)
        records = ctx.world.evidence_records(predicate)
        status = {Truth.TRUE: "verified", Truth.FALSE: "pending"}.get(truth, "unknown")
        obligations.append(ObligationStatus(predicate, status, records[-1].evidence_ids if records else ()))
    for scope in factory.history:
        decision = scope.result.decision if scope.result else None
        evidence = tuple(scope.result.evidence_ids) if scope.result else ()
        # No result means the skill never reached verification (backend failure/timeout).
        status = {Truth.TRUE: "verified", Truth.UNKNOWN: "unknown"}.get(decision, "failed")
        attempts.append(AttemptSummary(scope.attempt_id, status, (scope.expected_effect,), evidence, scope.skill,
                                       scope.arm))
    done = all(o.status == "verified" for o in obligations)
    reason = "completed" if done and run.goal_verified else "authored_tree_failed"
    report = EpisodeReport(run_id, scenario.goal.goal_id, reason, reason == "completed", tuple(obligations),
                           tuple(attempts), tuple(o.obligation_id for o in obligations if o.status == "verified"),
                           attempts_used=len(attempts), mode=mode)
    report.notes.append(f"authored task {task.name}: no planner, no memory ledger")
    ctx.world.log("episode_stop", run_id=run_id, mode=mode, reason=reason, completed=report.completed,
                  obligations={o.obligation_id: o.status for o in obligations})
    return report


def _run_exists(store: MemoryStore, run_id: str) -> bool:
    try:
        store.get_run(run_id)
        return True
    except KeyError:
        return False
