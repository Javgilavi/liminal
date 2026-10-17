"""V5 cooperative arms: a support arm maintains an observed condition while another arm works.

Mock arms and replay fixtures only: nothing here is evidence about hardware.
"""

import asyncio
import itertools
import time
from pathlib import Path

import pytest
import yaml

from liminal.app import build_context
from liminal.core.types import Skill, SkillResult, Status
from liminal.orchestration import bt
from liminal.orchestration.attempt import EvidenceAttempts
from liminal.orchestration.episode import run_authored
from liminal.orchestration.replay import (
    ReplayBackend,
    ReplayObserver,
    ReplayScene,
    attach_replay_primitives,
    load_scenario,
)
from liminal.orchestration.task_spec import load_task, parse_task
from liminal.perception.evidence import EvidenceRef, FactObservation, PredicateSpec, Truth
from liminal.safety import ArmConflict

ROBOT = "configs/robot/mock_dual.yaml"
TASK = Path("configs/tasks/replay_lid_hold_insert.yaml")


def run(coro):
    return asyncio.run(coro)


# --- replay end to end ---------------------------------------------------------------------------


def replay_run(tmp_path, name, task=None, backend=None, ttl=30.0):
    scenario = load_scenario(f"configs/replay/{name}.yaml")
    scene = ReplayScene(scenario, tmp_path / "scene.json")
    backend = backend or ReplayBackend(scene)
    ctx = build_context(ROBOT, backend="mock", policy_backend=backend)
    attach_replay_primitives(ctx, scene)
    observer = ReplayObserver(ctx.world, scene, ttl_s=ttl, bound=False)
    task = task or load_task(TASK, arms=ctx.arm_names)
    report = run(run_authored(scenario, ctx, task, observer))
    return report, ctx, scene, backend, observer


def kinds(ctx):
    return [e["kind"] for e in ctx.world.events]


def lid_observations(ctx, start_kind, end_kind, node):
    """lid_open observations recorded between an attempt starting and ending on `node`."""
    inside, out = False, []
    for e in ctx.world.events:
        if e["kind"] == start_kind and e.get("node") == node:
            inside = True
        elif e["kind"] == end_kind and e.get("node") == node:
            inside = False
        elif inside and e["kind"] == "observation" and e["predicate"] == "lid_open":
            out.append(e["value"])
    return out


def test_nominal_run_holds_lid_from_fresh_evidence_and_inserts(tmp_path):
    report, ctx, scene, backend, observer = replay_run(tmp_path, "lid_hold_insert")
    assert report.completed and [a.status for a in report.attempts] == ["verified", "verified"]
    assert [c.skill_id for c in backend.calls] == ["open_lid", "insert"]
    # Every lid_open observation taken while the insert ran said true, and there were several:
    # the monitor kept looking, nothing else would have.
    seen = lid_observations(ctx, "attempt_start", "attempt_end", "right:insert(block)")
    assert len(seen) >= 3 and set(seen) == {"true"}
    # The latent slip was suppressed only because the left arm was really holding.
    assert scene.fault_log[0]["outcome"] == "suppressed"
    assert "while_violated" not in kinds(ctx)
    # Ordered release came last; the lid then falls, the inserted block stays.
    assert scene.states["lid_open"] is False and scene.locations["block"] == "slot"


def test_slip_is_detected_from_evidence_recovered_and_completed(tmp_path):
    report, ctx, scene, backend, _ = replay_run(tmp_path, "lid_hold_insert_slip")
    assert report.completed
    assert [c.skill_id for c in backend.calls] == ["open_lid", "open_lid", "insert"]  # re-opened, insert cancelled
    ks = kinds(ctx)
    release = next(i for i, e in enumerate(ctx.world.events) if e.get("node") == "left:release")
    assert ks.index("while_violated") < ks.index("retry") < release
    assert [a.status for a in report.attempts] == ["verified", "failed", "verified", "verified"]
    assert backend.session("right").cancelled == 1 and backend.session("left").cancelled == 0
    assert ctx.arms["right"].stopped  # the body's arm was stopped through the SkillNode cancel path


def test_support_lost_after_n_seconds_is_detected_and_recovered(tmp_path):
    report, ctx, scene, backend, _ = replay_run(tmp_path, "lid_hold_insert_support_lost")
    assert report.completed and "while_violated" in kinds(ctx)
    assert [c.skill_id for c in backend.calls] == ["open_lid", "open_lid", "insert"]


def test_unrecoverable_slip_reports_failure_honestly(tmp_path):
    raw = yaml.safe_load(TASK.read_text())
    raw["tree"]["sequence"][0]["retry"]["attempts"] = 1
    report, ctx, scene, backend, _ = replay_run(tmp_path, "lid_hold_insert_slip", task=parse_task(raw))
    assert not report.completed and report.obligations[0].status == "pending"
    assert "hold_released" not in kinds(ctx)  # release is never reached after a violation
    assert ctx.arms["left"].holding is None  # the hold was dropped, not left fighting an unowned arm
    assert ctx.safety.owner_of("left") is None


def test_latent_slip_depends_on_a_real_hold_not_on_bookkeeping(tmp_path):
    """The fixture's physics: only an arm that is actually holding suppresses the latent slip."""
    scene = ReplayScene(load_scenario("configs/replay/lid_hold_insert.yaml"), tmp_path / "scene.json")
    scene._set_state("lid_open", True)
    fault = scene.begin_state_skill(Skill("right", "insert"))
    assert fault is not None and scene.slip(fault, set()) and scene.states["lid_open"] is False
    scene._set_state("lid_open", True)
    assert scene.begin_state_skill(Skill("right", "insert")) is None  # only the first insert is faulty
    scene.runs["insert"] = 0
    fault = scene.begin_state_skill(Skill("right", "insert"))
    assert not scene.slip(fault, {"left"}) and scene.states["lid_open"] is True


def test_per_arm_policy_state_is_separate(tmp_path):
    scenario = load_scenario("configs/replay/lid_hold_insert_slip.yaml")
    scene = ReplayScene(scenario, tmp_path / "scene.json")
    left, right = ReplayBackend(scene), ReplayBackend(scene)
    ctx = build_context(ROBOT, backend="mock", policy_backend=ReplayBackend(scene))
    ctx.skills.arm_policies = {"left": left, "right": right}  # option A2: one policy instance per arm
    attach_replay_primitives(ctx, scene)
    observer = ReplayObserver(ctx.world, scene, ttl_s=30.0, bound=False)
    report = run(run_authored(scenario, ctx, load_task(TASK, arms=ctx.arm_names), observer))
    assert report.completed
    assert {c.arm for c in left.calls} == {"left"} and {c.arm for c in right.calls} == {"right"}
    # Shared instance (A1): the sessions stay per arm all the same.
    shared = replay_run(tmp_path / "again", "lid_hold_insert")[3]
    assert set(shared.sessions) == {"left", "right"}
    assert [c.skill_id for c in shared.session("left").calls] == ["open_lid"]
    assert [c.skill_id for c in shared.session("right").calls] == ["insert"]


# --- While lifecycle with a self-contained evidence world ----------------------------------------


class Scene:
    def __init__(self):
        self.facts = {"lid_open": True, "block_in_slot": False}
        self.hidden: set[str] = set()
        self.refreshes = 0
        self.observer_broken = False


class Script:
    """Backend whose skills are async callables over the scene, recording who owns the left arm."""

    def __init__(self, scene, **steps):
        self.scene, self.steps, self.calls, self.left_owner = scene, steps, [], {}

    async def execute(self, skill, ctx):
        self.calls.append(skill.name)
        self.left_owner[skill.name] = ctx.safety.owner_of("left")
        await self.steps[skill.name](self.scene)
        return SkillResult(Status.SUCCESS, skill)


_ids = itertools.count(1)


def evidence_ctx(make_ctx, scene, backend, run_id="run-1", ttl=5.0):
    ctx = make_ctx(backend)
    world = ctx.world
    names = tuple(scene.facts)
    for p in names:
        world.register_predicate(PredicateSpec(p, ttl_s=ttl))
    writer = world.register_observer("scene", "1", set(names))
    world.begin_run(run_id)

    async def refresh():
        scene.refreshes += 1
        if scene.observer_broken:
            raise RuntimeError("camera offline")
        attempt = world.active_attempt_id or f"{run_id}:observe:{next(_ids)}"
        for p in names:
            n = next(_ids)
            value = Truth.UNKNOWN if p in scene.hidden else Truth(str(scene.facts[p]).lower())
            world.observe(FactObservation(
                predicate=p, value=value, evidence=(EvidenceRef(f"f{n}", camera_id="top", frame_seq=n),),
                observer_id="scene", observer_version="1", observed_utc="2026-10-17T12:00:00Z",
                observed_mono_s=time.monotonic(), run_id=run_id, attempt_id=attempt,
            ), writer)

    ctx.attempts = EvidenceAttempts(world, run_id, refresh, ("scene",))
    ctx.observe = refresh
    return ctx


async def sleep(seconds):
    await asyncio.sleep(seconds)


def coop(*, establish=True, release=True, period=0.02, body_skill=None, **kw):
    body = bt.SkillNode(body_skill or Skill("right", "insert", success="block_in_slot", verify_timeout_s=0.3))
    return bt.While(
        "left", "lid_open", body, until="block_in_slot", monitor_period_s=period, establish_timeout_s=0.3,
        establish=bt.SkillNode(Skill("left", "open_lid", success="lid_open", verify_timeout_s=0.3))
        if establish else None,
        release=bt.SkillNode(Skill("left", "release")) if release else None, **kw)


def good_steps(scene):
    async def open_lid(s):
        s.facts["lid_open"] = True

    async def insert(s):
        await sleep(0.08)
        s.facts["block_in_slot"] = True

    return {"open_lid": open_lid, "insert": insert}


def test_support_arm_is_claimed_for_the_whole_hold_lifetime(make_ctx):
    scene = Scene()
    scene.facts["lid_open"] = False
    steps = good_steps(scene)

    async def open_lid(s):
        s.facts["lid_open"] = True

    steps["open_lid"] = open_lid
    backend = Script(scene, **steps)
    ctx = evidence_ctx(make_ctx, scene, backend)
    node = coop(name="coop")
    owners = {}
    inner = ctx.skills.primitives

    class Probe:
        async def execute(self, skill, c):
            owners["release"] = c.safety.owner_of("left")
            return await inner.execute(skill, c)

    ctx.skills.primitives = Probe()
    assert run(node.run(ctx)) is Status.SUCCESS
    # establish, body-time and release all ran inside the While's single claim of the left arm
    assert backend.left_owner["open_lid"] == "coop" and owners["release"] == "coop"
    assert backend.left_owner["insert"] == "coop"
    assert ctx.safety.owner_of("left") is None and ctx.arms["left"].holding is None
    ks = kinds(ctx)
    assert ks.index("hold_engaged") < ks.index("hold_released")
    assert ctx.world.get("execution.left.hold_mode") is False


def test_body_on_the_support_arm_is_an_arm_conflict(make_ctx):
    scene = Scene()
    steps = good_steps(scene)
    steps["grab"] = steps["insert"]
    ctx = evidence_ctx(make_ctx, scene, Script(scene, **steps))
    node = coop(establish=False, release=False,
                body_skill=Skill("left", "grab", success="block_in_slot", verify_timeout_s=0.3))
    with pytest.raises(ArmConflict):
        run(node.run(ctx))
    assert ctx.arms["left"].holding is None and ctx.safety.owner_of("left") is None


def test_establish_and_release_must_be_support_arm_skill_nodes(make_ctx):
    body = bt.Check("x")
    with pytest.raises(ValueError, match="support arm"):
        bt.While("left", "lid_open", body, establish=bt.SkillNode(Skill("right", "open_lid")))
    with pytest.raises(ValueError, match="support arm"):
        bt.While("left", "lid_open", body, release=bt.Check("x"))


def test_unknown_evidence_is_loss_not_ok(make_ctx):
    scene = Scene()
    steps = good_steps(scene)

    async def insert(s):
        await sleep(0.05)
        s.hidden.add("lid_open")  # the camera loses the lid: unknown, not false
        await sleep(2.0)

    steps["insert"] = insert
    ctx = evidence_ctx(make_ctx, scene, Script(scene, **steps))
    t0 = time.monotonic()
    assert run(coop().run(ctx)) is Status.FAILURE
    assert time.monotonic() - t0 < 1.0  # cancelled long before the body's 2 s
    assert [e for e in ctx.world.events if e["kind"] == "while_violated"]
    assert ctx.arms["right"].stopped


def test_unknown_at_establishment_is_not_established(make_ctx):
    scene = Scene()
    scene.hidden.add("lid_open")
    ctx = evidence_ctx(make_ctx, scene, Script(scene, **good_steps(scene)))
    assert run(coop(establish=False).run(ctx)) is Status.FAILURE
    assert "while_not_established" in kinds(ctx)
    assert ctx.arms["left"].holding is None


def test_a_failing_observer_counts_as_loss(make_ctx):
    scene = Scene()
    steps = good_steps(scene)

    async def insert(s):
        await sleep(0.05)
        s.observer_broken = True
        await sleep(2.0)

    steps["insert"] = insert
    ctx = evidence_ctx(make_ctx, scene, Script(scene, **steps))
    assert run(coop().run(ctx)) is Status.FAILURE
    assert "while_observe_failed" in kinds(ctx)


def test_expiry_without_any_refresh_is_loss(make_ctx):
    """Outside evidence mode nobody refreshes; an observation that expires is loss of sight."""
    scene = Scene()

    async def insert(s):
        await sleep(2.0)

    ctx = evidence_ctx(make_ctx, scene, Script(scene, insert=insert), ttl=0.15)
    run(ctx.observe())  # one observation, then silence
    ctx.observe, ctx.attempts = None, None
    node = bt.While("left", "lid_open", bt.SkillNode(Skill("right", "insert")), establish_timeout_s=0.3)
    t0 = time.monotonic()
    assert run(node.run(ctx)) is Status.FAILURE
    assert time.monotonic() - t0 < 1.0


def test_periodic_refresh_catches_a_slip_that_nothing_else_would(make_ctx):
    scene = Scene()
    steps = good_steps(scene)

    async def insert(s):
        await sleep(0.05)
        s.facts["lid_open"] = False  # slips silently; the body keeps going
        await sleep(2.0)

    steps["insert"] = insert
    ctx = evidence_ctx(make_ctx, scene, Script(scene, **steps), ttl=30.0)  # no expiry to save us
    refreshes = scene.refreshes
    t0 = time.monotonic()
    assert run(coop().run(ctx)) is Status.FAILURE
    assert time.monotonic() - t0 < 1.0 and scene.refreshes - refreshes >= 3


def test_simultaneous_completion_cannot_hide_a_violation(make_ctx):
    scene = Scene()
    steps = good_steps(scene)

    async def insert(s):
        await sleep(0.05)
        s.facts["block_in_slot"] = True  # the body really succeeds ...
        s.facts["lid_open"] = False  # ... and the lid falls in the same instant

    steps["insert"] = insert
    ctx = evidence_ctx(make_ctx, scene, Script(scene, **steps))
    # No periodic refresh at all (period far beyond the run): only the post-body look can see it.
    node = coop(period=60.0, release=True)
    assert run(node.run(ctx)) is Status.FAILURE
    assert "while_violated" in kinds(ctx) and "hold_released" not in kinds(ctx)
    insert_attempt = next(a for a in ctx.attempts.history if a.expected_effect == "block_in_slot")
    # The partial two-arm commit fails the attempt: the body's own verification is voided.
    assert insert_attempt.result.decision is Truth.FALSE and insert_attempt.voided
    assert any(e["kind"] == "attempt_voided" for e in ctx.world.events)


def test_body_success_with_lid_held_then_releases_in_order(make_ctx):
    scene = Scene()
    ctx = evidence_ctx(make_ctx, scene, Script(scene, **good_steps(scene)))
    assert run(coop().run(ctx)) is Status.SUCCESS
    ks = kinds(ctx)
    last_insert_end = max(i for i, e in enumerate(ctx.world.events)
                          if e["kind"] == "attempt_end" and "insert" in e["node"])
    release_start = next(i for i, e in enumerate(ctx.world.events)
                         if e["kind"] == "node_start" and e["node"] == "left:release")
    assert last_insert_end < release_start < ks.index("hold_released")
    assert [a.expected_effect for a in ctx.attempts.history] == ["lid_open", "block_in_slot"]
    assert len({a.attempt_id for a in ctx.attempts.history}) == 2  # establish and body: own attempts


def test_legacy_while_without_release_keeps_the_hold_engaged(make_ctx):
    scene = Scene()
    ctx = evidence_ctx(make_ctx, scene, Script(scene, **good_steps(scene)))
    assert run(coop(establish=False, release=False).run(ctx)) is Status.SUCCESS
    assert ctx.arms["left"].holding is not None and ctx.safety.owner_of("left") is None
    assert ctx.world.get("execution.left.hold_mode") is True


def test_hold_mode_is_not_evidence_of_support(make_ctx):
    scene = Scene()
    scene.facts["lid_open"] = False  # nobody opened the lid
    ctx = evidence_ctx(make_ctx, scene, Script(scene, **good_steps(scene)))
    assert run(coop(establish=False).run(ctx)) is Status.FAILURE  # engaged hold does not establish it
    assert not scene.facts["block_in_slot"]
