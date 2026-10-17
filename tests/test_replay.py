"""Replay layer tests: mock-only, no hardware, network or model."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from liminal.core.types import Skill, Status
from liminal.orchestration.plan_validation import PlanningContext, validate_proposal
from liminal.orchestration.replay import (
    ReplayBackend,
    ReplayModel,
    ReplayObserver,
    ReplayScene,
    load_scenario,
    planning_frames,
)
from liminal.perception.evidence import Truth
from liminal.perception.verifier import VerificationRequest, Verifier
from liminal.perception.world_state import WorldState

CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "replay"
SCENARIOS = sorted(CONFIGS.glob("*.yaml"))


def _skill(obj: str = "red", dest: str = "box", arm: str = "left") -> Skill:
    return Skill(arm, "pick_place", prompt="x", target=obj, success=f"{obj}_in_{dest}",
                 params={"object_id": obj, "destination_id": dest})


def _scene(tmp_path: Path, name: str = "two_blocks") -> ReplayScene:
    return ReplayScene(load_scenario(CONFIGS / f"{name}.yaml"), tmp_path / "scene.json")


def _raw(name: str = "two_blocks") -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _context(scene: ReplayScene, completed: tuple[str, ...] = ()) -> PlanningContext:
    scenario = scene.scenario
    snapshot = scene.capture()
    return PlanningContext(
        "run-1", snapshot.revision, scenario.goal, planning_frames(snapshot), (), scenario.objects,
        scenario.registry,
        progress={"completed_effects": list(completed),
                  "pending_effects": [p for p in scenario.goal.obligations if p not in completed]},
    )


# --- scenario loading -------------------------------------------------------------------------

def test_shipped_scenarios_load():
    assert {path.stem for path in SCENARIOS} >= {"two_blocks", "occluded_reference", "moved_destination"}
    for path in SCENARIOS:
        scenario = load_scenario(path)
        assert scenario.name == path.stem
        assert set(scenario.obligations) == set(scenario.goal.obligations)


def test_two_blocks_has_two_obligations():
    assert len(load_scenario(CONFIGS / "two_blocks.yaml").goal.obligations) == 2


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(bogus=1),
    lambda r: r["obligations"].pop("red_in_box"),
    lambda r: r["obligations"]["red_in_box"].update(destination_id="nowhere"),
    lambda r: r["obligations"]["red_in_box"].update(destination_id="red"),
    lambda r: r["capabilities"].update(available_arms=[]),
    lambda r: r["capabilities"]["skills"]["pick_place"].update(nope=1),
    lambda r: r.update(initial_locations={"red": "nowhere"}),
    lambda r: r.update(reference_frames={"f": "ghost"}),
    lambda r: r.update(initially_occluded=["box"]),
    lambda r: r["goal"].pop("text"),
])
def test_invalid_scenarios_rejected(tmp_path, mutate):
    raw = _raw()
    mutate(raw)
    with pytest.raises(ValueError):
        load_scenario(_write(tmp_path, raw))


def test_non_mapping_scenario_rejected(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("- a\n")
    with pytest.raises(ValueError):
        load_scenario(path)


# --- scene ------------------------------------------------------------------------------------

def test_apply_changes_location_and_revision(tmp_path):
    scene = _scene(tmp_path)
    assert scene.revision == 0
    assert scene.capture().facts == {"red_in_box": Truth.FALSE, "blue_in_box": Truth.FALSE}
    assert scene.apply(_skill("red")) == 1
    assert scene.locations["red"] == "box"
    assert scene.capture().facts["red_in_box"] is Truth.TRUE
    assert scene.apply(_skill("red")) == 1  # already there: no revision bump


@pytest.mark.parametrize("skill", [
    Skill("left", "unknown_skill", target="red", params={"object_id": "red", "destination_id": "box"}),
    Skill("arm9", "pick_place", target="red", params={"object_id": "red", "destination_id": "box"}),
    Skill("left", "pick_place", target="red", params={"object_id": "red"}),
    Skill("left", "pick_place", target="blue", params={"object_id": "red", "destination_id": "box"}),
    Skill("left", "pick_place", target="red", params={"object_id": "red", "destination_id": "red"}),
])
def test_apply_rejects_bad_skills(tmp_path, skill):
    scene = _scene(tmp_path)
    with pytest.raises(ValueError):
        scene.apply(skill)
    assert scene.revision == 0


def test_state_persists_across_reopen(tmp_path):
    scene = _scene(tmp_path)
    scene.apply(_skill("red"))
    reopened = _scene(tmp_path)
    assert reopened.revision == 1
    assert reopened.locations == {"red": "box", "blue": "table"}
    assert reopened.capture().facts["red_in_box"] is Truth.TRUE


def test_reopen_with_wrong_scenario_or_corrupt_state(tmp_path):
    _scene(tmp_path)
    with pytest.raises(ValueError):
        ReplayScene(load_scenario(CONFIGS / "moved_destination.yaml"), tmp_path / "scene.json")
    (tmp_path / "scene.json").write_text("{not json")
    with pytest.raises(ValueError):
        _scene(tmp_path)


def test_snapshot_images_are_png_and_reference_is_stable(tmp_path):
    scene = _scene(tmp_path, "occluded_reference")
    first, second = scene.capture(), scene.capture()
    assert all(image.data.startswith(b"\x89PNG") for image in first.images)
    current = [i for i in first.images if i.event == "schematic_current_scene"]
    reference = [i for i in first.images if i.event.startswith("goal_reference:")]
    assert len(current) == 1 and len(reference) == 1 and reference[0].event == "goal_reference:red"
    assert reference[0].frame_id == [i for i in second.images if i.event.startswith("goal_")][0].frame_id
    assert current[0].frame_id != [i for i in second.images if i.event == "schematic_current_scene"][0].frame_id
    assert reference[0].data != current[0].data  # target visible in reference, hidden now


def test_relocate_undoes_completion(tmp_path):
    scene = _scene(tmp_path, "moved_destination")
    scene.apply(_skill("red", "left_bin"))
    assert scene.capture().facts["red_in_left_bin"] is Truth.TRUE
    scene.relocate("red", "right_bin")
    assert scene.capture().facts["red_in_left_bin"] is Truth.FALSE
    with pytest.raises(ValueError):
        scene.relocate("red", "ghost")
    with pytest.raises(ValueError):
        scene.relocate("left_bin", "table")


# --- backend ----------------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def test_backend_normal_applies(tmp_path):
    scene = _scene(tmp_path)
    backend = ReplayBackend(scene)
    result = _run(backend.execute(_skill("red"), None))
    assert result.status is Status.SUCCESS and scene.revision == 1
    call = backend.calls[0]
    assert (call.revision_before, call.revision_after) == (0, 1)


def test_backend_fail_first_leaves_scene_unchanged(tmp_path):
    scene = _scene(tmp_path)
    backend = ReplayBackend(scene, fail_first=True)
    assert _run(backend.execute(_skill("red"), None)).status is Status.FAILURE
    assert scene.revision == 0
    assert _run(backend.execute(_skill("red"), None)).status is Status.SUCCESS
    assert scene.revision == 1


def test_backend_lie_first_claims_success_without_change(tmp_path):
    scene = _scene(tmp_path)
    backend = ReplayBackend(scene, lie_first=True)
    assert _run(backend.execute(_skill("red"), None)).status is Status.SUCCESS
    assert scene.revision == 0 and scene.locations["red"] == "table"
    _run(backend.execute(_skill("red"), None))
    assert scene.locations["red"] == "box"


def test_backend_fail_then_lie(tmp_path):
    scene = _scene(tmp_path)
    backend = ReplayBackend(scene, fail_first=True, lie_first=True)
    statuses = [_run(backend.execute(_skill("red"), None)).status for _ in range(3)]
    assert statuses == [Status.FAILURE, Status.SUCCESS, Status.SUCCESS]
    assert [c.revision_after for c in backend.calls] == [0, 0, 1]


# --- observer ---------------------------------------------------------------------------------

def _observer(scene: ReplayScene) -> tuple[WorldState, ReplayObserver]:
    world = WorldState()
    observer = ReplayObserver(world, scene)
    world.begin_run("run-1")
    world.begin_attempt("att-1")
    observer.bind("run-1", "att-1")
    return world, observer


def test_observer_true_false(tmp_path):
    scene = _scene(tmp_path)
    world, observer = _observer(scene)
    red = {"object_id": "red", "destination_id": "box"}
    blue = {"object_id": "blue", "destination_id": "box"}
    assert world.evaluate("red_in_box", bindings=red) is Truth.UNKNOWN  # nothing observed yet
    _run(observer.refresh())
    assert world.evaluate("red_in_box", bindings=red) is Truth.FALSE
    scene.apply(_skill("red"))
    _run(observer.refresh())
    assert world.evaluate("red_in_box", bindings=red) is Truth.TRUE
    assert world.evaluate("blue_in_box", bindings=blue) is Truth.FALSE
    assert observer.refresh_count == 2


def test_observer_occluded_is_unknown_not_false(tmp_path):
    scene = _scene(tmp_path, "occluded_reference")
    world, observer = _observer(scene)
    red = {"object_id": "red", "destination_id": "box"}
    _run(observer.refresh())
    assert world.evaluate("red_in_box", bindings=red) is Truth.UNKNOWN
    assert not world.holds("red_in_box", bindings=red)
    scene.reveal("red")
    _run(observer.refresh())
    assert world.evaluate("red_in_box", bindings=red) is Truth.FALSE
    scene.occlude("red")
    _run(observer.refresh())
    assert world.evaluate("red_in_box", bindings=red) is Truth.UNKNOWN


def test_observer_cannot_set_physical_facts_directly(tmp_path):
    scene = _scene(tmp_path)
    world, _ = _observer(scene)
    with pytest.raises(PermissionError):
        world.set("red_in_box", True)


def test_lying_backend_is_caught_by_verification(tmp_path):
    scene = _scene(tmp_path)
    world, observer = _observer(scene)
    backend = ReplayBackend(scene, lie_first=True)
    bindings = {"object_id": "red", "destination_id": "box"}

    async def attempt() -> Truth:
        sequence = world.sequence
        result = await backend.execute(_skill("red"), None)
        assert result.status is Status.SUCCESS  # the claim
        make = observer.make_refresh("run-1", "att-1")
        boundary = world.monotonic_now()
        await make()
        verdict = await Verifier().verify(VerificationRequest(
            "run-1", "att-1", "red_in_box", bindings, sequence, (ReplayObserver.OBSERVER_ID,),
            world.monotonic_now() + 1.0, evidence_after_mono_s=boundary), world)
        return verdict.decision

    assert _run(attempt()) is Truth.FALSE


def test_honest_backend_verifies_true(tmp_path):
    scene = _scene(tmp_path)
    world, observer = _observer(scene)
    backend = ReplayBackend(scene)
    bindings = {"object_id": "red", "destination_id": "box"}

    async def attempt() -> Truth:
        sequence = world.sequence
        await backend.execute(_skill("red"), None)
        boundary = world.monotonic_now()
        await observer.refresh()
        verdict = await Verifier().verify(VerificationRequest(
            "run-1", "att-1", "red_in_box", bindings, sequence, (ReplayObserver.OBSERVER_ID,),
            world.monotonic_now() + 1.0, evidence_after_mono_s=boundary), world)
        return verdict.decision

    assert _run(attempt()) is Truth.TRUE


def test_observer_requires_ids(tmp_path):
    scene = _scene(tmp_path)
    observer = ReplayObserver(WorldState(), scene)
    with pytest.raises(ValueError):
        _run(observer.refresh())


# --- model ------------------------------------------------------------------------------------

def _plan(scene: ReplayScene, completed: tuple[str, ...] = ()):
    from liminal.orchestration.planner import ConstrainedPlanner

    model = ReplayModel(scene.scenario)
    context = _context(scene, completed)
    return ConstrainedPlanner(model).plan(context), model


@pytest.mark.parametrize("name", [p.stem for p in SCENARIOS])
def test_model_output_validates_for_each_scenario(tmp_path, name):
    scene = _scene(tmp_path, name)
    context = _context(scene)
    model = ReplayModel(scene.scenario)
    proposal = json.loads(model.complete_context("sys", _user(context), context.observations))
    plan = validate_proposal(proposal, context)
    first = scene.scenario.goal.obligations[0]
    assert plan.task.goal == first
    assert plan.used_evidence


def _user(context: PlanningContext) -> str:
    from liminal.orchestration.planner import ConstrainedPlanner

    return ConstrainedPlanner(ReplayModel(load_scenario(CONFIGS / "two_blocks.yaml")))._prompt(context)


def test_model_never_proposes_completed_obligation(tmp_path):
    scene = _scene(tmp_path)
    plan, _ = _plan(scene, completed=("red_in_box",))
    assert plan.task.goal == "blue_in_box"
    plan, _ = _plan(scene, completed=("blue_in_box",))
    assert plan.task.goal == "red_in_box"


def test_model_without_pending_effects_falls_back_to_goal(tmp_path):
    scene = _scene(tmp_path)
    scenario = scene.scenario
    snapshot = scene.capture()
    context = PlanningContext("run-1", 0, scenario.goal, planning_frames(snapshot), (), scenario.objects,
                              scenario.registry, progress={"completed_effects": ["red_in_box"]})
    model = ReplayModel(scenario)
    proposal = json.loads(model.complete("s", _user(context)))
    assert proposal["steps"][0]["effects"] == ["blue_in_box"]


def test_model_with_everything_completed_raises(tmp_path):
    scene = _scene(tmp_path)
    context = _context(scene, completed=("red_in_box", "blue_in_box"))
    with pytest.raises(ValueError):
        ReplayModel(scene.scenario).complete("s", _user(context))


def test_planning_frames_roles(tmp_path):
    scene = _scene(tmp_path, "occluded_reference")
    frames = planning_frames(scene.capture())
    assert [f.temporal_role for f in frames] == ["historical", "current"]
    assert frames[-1].revision == scene.revision and frames[0].media_type == "image/png"


def test_refresh_uses_world_ids_and_observation_only_ids(tmp_path):
    scene = _scene(tmp_path)
    world = WorldState()
    observer = ReplayObserver(world, scene)  # never bound
    world.begin_run("run-9")
    world.begin_attempt("att-9")
    _run(observer.refresh())
    red = {"object_id": "red", "destination_id": "box"}
    record = world.evidence_records("red_in_box", red)[-1]
    assert record.observation.attempt_id == "att-9"
    world.end_attempt()
    _run(observer.refresh())  # outside any attempt: fresh observation-only id
    _run(observer.refresh())
    ids = [r.observation.attempt_id for r in world.evidence_records("red_in_box", red)]
    assert ids[1:] == ["run-9:observe:1", "run-9:observe:2"]
    assert world.evaluate("red_in_box", bindings=red) is Truth.FALSE


def test_scheduled_events_fire_once_after_enough_actions_and_persist(tmp_path):
    scenario = load_scenario(CONFIGS / "moved_destination.yaml")
    assert [(e.after_actions, e.kind, e.object_id) for e in scenario.events] == [(1, "relocate", "red")]
    scene = ReplayScene(scenario, tmp_path / "s.json")
    assert scene.fire_due_events() == []  # nothing has been placed yet
    scene.relocate("red", "left_bin")
    scene.actions = 1
    assert [e.object_id for e in scene.fire_due_events()] == ["red"]
    assert scene.locations["red"] == "table"
    assert scene.fire_due_events() == []  # fires once
    assert ReplayScene(scenario, tmp_path / "s.json").events_fired == 1  # and survives a restart


def test_bad_scheduled_events_are_rejected(tmp_path):
    import yaml

    raw = yaml.safe_load((CONFIGS / "moved_destination.yaml").read_text())
    for bad in ({"after_actions": 0, "kind": "relocate", "object_id": "red"},
                {"after_actions": 1, "kind": "teleport", "object_id": "red"},
                {"after_actions": 1, "kind": "relocate", "object_id": "left_bin"},
                {"after_actions": 1, "kind": "relocate", "object_id": "red", "location": "nowhere"}):
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump({**raw, "events": [bad]}))
        with pytest.raises(ValueError):
            load_scenario(path)


# --- V5 cooperative state (scenario `state:` block) ---------------------------------------------

LID = "lid_hold_insert"


def test_object_scenarios_keep_their_exact_saved_format_and_facts(tmp_path):
    scene = _scene(tmp_path)
    snapshot = scene.capture()
    assert set(snapshot.facts) == set(scene.scenario.obligations)
    assert "states" not in json.loads((tmp_path / "scene.json").read_text())


def test_cooperative_scene_exposes_observed_state_and_persists_it(tmp_path):
    scene = _scene(tmp_path, LID)
    snapshot = scene.capture()
    assert snapshot.facts["lid_open"] is Truth.FALSE and snapshot.bindings["lid_open"] == {}
    scene._set_state("lid_open", True)
    reloaded = ReplayScene(load_scenario(CONFIGS / f"{LID}.yaml"), tmp_path / "scene.json")
    assert reloaded.states == {"lid_open": True} and reloaded.capture().facts["lid_open"] is Truth.TRUE


def test_state_skill_effects_follow_physics(tmp_path):
    scene = _scene(tmp_path, LID)
    insert = Skill("right", "insert", target="block", success="block_in_slot",
                   params={"object_id": "block", "destination_id": "slot"})
    assert not scene.complete_state_skill(insert) and scene.locations["block"] == "table"  # lid closed
    assert scene.complete_state_skill(Skill("left", "open_lid", target="lid", success="lid_open"))
    assert scene.complete_state_skill(insert) and scene.locations["block"] == "slot"
    scene.support_released("left")
    assert scene.states["lid_open"] is False


def test_support_lost_fires_lazily_and_once(tmp_path):
    scene = _scene(tmp_path, "lid_hold_insert_support_lost")
    scene._set_state("lid_open", True)
    scene._true_since["lid_open"] -= 1.0
    assert scene.capture().facts["lid_open"] is Truth.FALSE
    scene._set_state("lid_open", True)
    scene._true_since["lid_open"] -= 1.0
    assert scene.capture().facts["lid_open"] is Truth.TRUE  # a one-shot fault


def test_observer_registers_state_predicates_and_writes_them(tmp_path):
    scene = _scene(tmp_path, LID)
    world = WorldState()
    observer = ReplayObserver(world, scene, ttl_s=30.0, bound=False)
    world.begin_run("r")
    asyncio.run(observer.refresh())
    assert world.evaluate("lid_open") is Truth.FALSE and world.evaluate("block_in_slot") is Truth.FALSE


@pytest.mark.parametrize("mutate", [
    lambda r: r["state"]["predicates"].update(block_in_slot=False),
    lambda r: r["state"]["predicates"].update(lid_open="yes"),
    lambda r: r["state"]["skills"].update(teleport={"duration_s": 1}),
    lambda r: r["state"]["skills"]["insert"].update(requires=["nope"]),
    lambda r: r["state"]["support"]["lid_open"].update(arm="third"),
    lambda r: r["state"]["faults"].append({"kind": "gremlin", "state": "lid_open"}),
    lambda r: r["state"]["faults"].append({"kind": "slip_during", "state": "lid_open", "skill": "home"}),
    lambda r: r["state"].update(bogus=1),
])
def test_bad_state_blocks_are_rejected(tmp_path, mutate):
    raw = _raw(LID)
    mutate(raw)
    with pytest.raises(ValueError):
        load_scenario(_write(tmp_path, raw))


def test_distinct_destinations_render_in_distinct_regions(tmp_path):
    from liminal.orchestration.replay import _render_png

    scenario = load_scenario(CONFIGS / "workbench.yaml")

    def pixels(locations):
        import struct
        import zlib

        png = _render_png(scenario, locations)
        pos, idat = 8, b""
        while pos < len(png):
            n = struct.unpack(">I", png[pos:pos + 4])[0]
            if png[pos + 4:pos + 8] == b"IDAT":
                idat += png[pos + 8:pos + 8 + n]
            pos += 12 + n
        raw = zlib.decompress(idat)
        return [raw[1 + y * 961:(y + 1) * 961] for y in range(180)]

    def where(rows, color):
        xs = [x for row in rows for x in range(320) if tuple(row[3 * x:3 * x + 3]) == color]
        return (min(xs), max(xs)) if xs else None

    green = (60, 170, 80)
    base = {k: "table" for k in ("red", "blue", "green")}
    assert where(pixels(base), green) is not None
    left = where(pixels({**base, "green": "left_bin"}), green)
    right = where(pixels({**base, "green": "right_bin"}), green)
    assert left[1] < right[0]  # left bin is left of right bin, objects inside their own container
