"""V1: evidence-backed authored procedures, one fresh attempt per verified skill run."""

import asyncio
import itertools
import time

import pytest

from liminal.core.types import Skill, SkillResult, Status
from liminal.orchestration import bt
from liminal.orchestration.attempt import EvidenceAttempts
from liminal.orchestration.executor import run_task
from liminal.orchestration.task_spec import parse_task
from liminal.perception.evidence import EvidenceRef, FactObservation, PredicateSpec, Truth

PREDICATES = ("red_in_box", "blue_in_box")
_frames = itertools.count(1)


class Scene:
    """Ground truth the observer reads. Backends may lie; the scene does not."""

    def __init__(self):
        self.facts = {p: False for p in PREDICATES}
        self.hidden: set[str] = set()


class SceneBackend:
    def __init__(self, scene, *, lie_on=(), fail_on=()):
        self.scene, self.lie_on, self.fail_on = scene, list(lie_on), list(fail_on)
        self.calls: list[str] = []

    async def execute(self, skill, ctx):
        self.calls.append(skill.name)
        n = self.calls.count(skill.name)
        await asyncio.sleep(0.005)
        if (skill.name, n) in self.fail_on:
            return SkillResult(Status.FAILURE, skill, "grasp missed")
        if skill.success and (skill.name, n) not in self.lie_on:
            self.scene.facts[skill.success] = True
        return SkillResult(Status.SUCCESS, skill, "policy says done")


def evidence_world(ctx, scene, run_id="run-1"):
    world = ctx.world
    for p in PREDICATES:
        world.register_predicate(PredicateSpec(p, ttl_s=5.0))
    writer = world.register_observer("scene", "1", set(PREDICATES))
    world.begin_run(run_id)

    async def refresh():
        attempt = world.active_attempt_id or f"{run_id}:observe:{next(_frames)}"
        for p in PREDICATES:
            frame = next(_frames)
            value = Truth.UNKNOWN if p in scene.hidden else Truth(str(scene.facts[p]).lower())
            world.observe(FactObservation(
                predicate=p, value=value, evidence=(EvidenceRef(f"f{frame}", camera_id="top", frame_seq=frame),),
                observer_id="scene", observer_version="1", observed_utc="2026-10-17T12:00:00Z",
                observed_mono_s=time.monotonic(), run_id=run_id, attempt_id=attempt,
            ), writer)

    ctx.attempts = EvidenceAttempts(world, run_id, refresh, ("scene",))
    ctx.observe = refresh
    return ctx


def two_block_task(retries=1):
    def place(color):
        skill = {"skill": {"arm": "right", "name": f"place_{color}", "success": f"{color}_in_box",
                           "verify_timeout_s": 0.3}}
        guarded = {"fallback": [{"check": f"{color}_in_box"}, skill]}
        return {"retry": {"attempts": retries, "do": guarded}} if retries > 1 else guarded

    return parse_task({"name": "two_blocks", "goal": "blue_in_box",
                       "tree": {"sequence": [place("red"), place("blue")]}})


def test_authored_procedure_verifies_each_skill_with_its_own_attempt(make_ctx):
    scene = Scene()
    backend = SceneBackend(scene)
    ctx = evidence_world(make_ctx(backend), scene)
    report = asyncio.run(run_task(two_block_task(), ctx))
    assert report.goal_verified and report.status is Status.SUCCESS
    attempts = ctx.attempts.history
    assert [a.expected_effect for a in attempts] == ["red_in_box", "blue_in_box"]
    assert len({a.attempt_id for a in attempts}) == 2
    assert all(a.result.decision is Truth.TRUE and a.result.evidence_ids for a in attempts)


def test_backend_success_claim_does_not_advance_progress(make_ctx):
    scene = Scene()
    backend = SceneBackend(scene, lie_on=[("place_red", 1)])
    ctx = evidence_world(make_ctx(backend), scene)
    report = asyncio.run(run_task(two_block_task(), ctx))
    assert not report.goal_verified
    assert backend.calls == ["place_red"]  # blue never starts after an unverified red
    assert ctx.attempts.history[0].result.decision is Truth.FALSE


def test_retry_gets_a_fresh_attempt_and_recovers(make_ctx):
    scene = Scene()
    backend = SceneBackend(scene, lie_on=[("place_red", 1)], fail_on=[("place_blue", 1)])
    ctx = evidence_world(make_ctx(backend), scene)
    report = asyncio.run(run_task(two_block_task(retries=2), ctx))
    assert report.goal_verified
    assert backend.calls == ["place_red", "place_red", "place_blue", "place_blue"]
    red = [a for a in ctx.attempts.history if a.expected_effect == "red_in_box"]
    assert [a.result.decision for a in red] == [Truth.FALSE, Truth.TRUE]
    assert red[0].attempt_id != red[1].attempt_id


def test_current_state_check_skips_work_already_done(make_ctx):
    scene = Scene()
    scene.facts["red_in_box"] = True
    backend = SceneBackend(scene)
    ctx = evidence_world(make_ctx(backend), scene)
    report = asyncio.run(run_task(two_block_task(), ctx))
    assert report.goal_verified
    assert backend.calls == ["place_blue"]


def test_unknown_evidence_never_verifies(make_ctx):
    scene = Scene()
    scene.hidden.add("red_in_box")  # occluded: the placement happened but cannot be seen
    backend = SceneBackend(scene)
    ctx = evidence_world(make_ctx(backend), scene)
    report = asyncio.run(run_task(two_block_task(), ctx))
    assert not report.goal_verified
    assert ctx.attempts.history[0].result.decision is Truth.UNKNOWN


def test_goal_must_be_freshly_observed(make_ctx):
    scene = Scene()
    backend = SceneBackend(scene)
    ctx = evidence_world(make_ctx(backend), scene)
    task = two_block_task()
    task.goal = "red_in_box"

    async def run():
        status = await task.tree.run(ctx)
        scene.facts["red_in_box"] = False  # knocked out after it was verified
        return status, await run_task(parse_task({"name": "noop", "goal": "red_in_box",
                                                  "tree": {"check": "blue_in_box"}}), ctx)

    status, report = asyncio.run(run())
    assert status is Status.SUCCESS
    assert not report.goal_verified


def test_evidence_backed_tree_rejects_initial_state_and_unnamed_goal(make_ctx):
    scene = Scene()
    ctx = evidence_world(make_ctx(SceneBackend(scene)), scene)
    forged = parse_task({"name": "forged", "initial_state": {"red_in_box": True}, "tree": {"check": "red_in_box"}})
    with pytest.raises(ValueError, match="initialize physical facts"):
        asyncio.run(run_task(forged, ctx))
    unnamed = parse_task({"name": "unnamed", "tree": {"wait": {"condition": "x", "timeout_s": 0.01}}})
    assert not asyncio.run(run_task(unnamed, ctx)).goal_verified


def test_skill_without_success_predicate_runs_outside_an_attempt(make_ctx):
    scene = Scene()
    backend = SceneBackend(scene)
    ctx = evidence_world(make_ctx(backend), scene)
    status = asyncio.run(bt.SkillNode(Skill(arm="left", name="home")).run(ctx))
    assert status is Status.SUCCESS and ctx.attempts.history == []


def test_void_since_turns_verified_attempts_into_failures(make_ctx):
    scene = Scene()
    ctx = evidence_world(make_ctx(SceneBackend(scene)), scene)
    asyncio.run(run_task(two_block_task(), ctx))
    history = ctx.attempts.history
    assert all(a.result.decision is Truth.TRUE for a in history)
    assert ctx.attempts.void_since(1, "support lost") == [history[1].attempt_id]
    assert history[0].result.decision is Truth.TRUE and history[0].voided is None
    assert history[1].result.decision is Truth.FALSE and history[1].voided == "support lost"
    assert ctx.attempts.void_since(1, "again") == []  # already a failure: nothing left to void
