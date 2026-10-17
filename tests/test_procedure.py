"""Procedure contract: closed-set validation, trusted rendering, grounding and compilation."""

import asyncio

import pytest

from liminal.app import build_context
from liminal.orchestration.episode import run_authored
from liminal.orchestration.procedure import (
    KeywordGroundingModel,
    ProcedureError,
    SkillWorld,
    compile_procedure,
    describe,
    ground_goal,
    parse_procedure,
    scenario_with_procedure,
)
from liminal.orchestration.replay import ReplayBackend, ReplayObserver, ReplayScene, load_scenario

WORKBENCH = "configs/replay/workbench.yaml"


@pytest.fixture
def world():
    s = load_scenario(WORKBENCH)
    return SkillWorld(s.objects, s.registry)


def step(**kw):
    return {"id": "s1", "skill": "pick_place", "object_id": "red", "destination_id": "box", **kw}


def test_parse_renders_trusted_effect_and_arm(world):
    p = parse_procedure({"steps": [step()], "obligations": ["red_in_box"]}, world)
    assert p.steps[0].effect == "red_in_box" and p.steps[0].arm == "left"
    row = describe(p, world)[0]
    assert row["flux_prompt"] == "Pick up the red block and place it inside the white box."


@pytest.mark.parametrize("bad, match", [
    ({"steps": [step(skill="teleport")], "obligations": ["red_in_box"]}, "unregistered skill"),
    ({"steps": [step(object_id="unicorn")], "obligations": []}, "unknown object"),
    ({"steps": [step(effect="blue_in_box")], "obligations": ["blue_in_box"]}, "registered effect"),
    ({"steps": [step(arm="third")], "obligations": ["red_in_box"]}, "cannot run"),
    ({"steps": [step(depends_on=["s9"])], "obligations": ["red_in_box"]}, "depends_on"),
    ({"steps": [step()], "obligations": ["green_in_box"]}, "not produced"),
    ({"steps": [step(), step(id="s2")], "obligations": ["red_in_box"]}, "more than one step"),
    ({"steps": [step(code="import os")], "obligations": ["red_in_box"]}, "unknown fields"),
    ({"steps": [step()], "obligations": ["red_in_box"], "initial_state": {}}, "unknown fields"),
])
def test_parse_rejects_anything_outside_the_closed_sets(world, bad, match):
    with pytest.raises(ProcedureError, match=match):
        parse_procedure(bad, world)


@pytest.mark.parametrize("goal, expected, arms", [
    ("Put the red block in the box", ["red_in_box"], ["left"]),
    ("put the red and blue blocks in the box", ["red_in_box", "blue_in_box"], ["left", "left"]),
    ("Put the green block in the left bin, then the red block in the right bin with the right arm",
     ["green_in_left_bin", "red_in_right_bin"], ["left", "right"]),
])
def test_keyword_grounding(world, goal, expected, arms):
    p = ground_goal(goal, world, KeywordGroundingModel(world))
    assert list(p.obligations) == expected
    assert [s.arm for s in p.steps] == arms


def test_grounding_reports_unsupported_instead_of_inventing(world):
    p = ground_goal("stack the blue block on the red block, then put the green block in the box", world,
                    KeywordGroundingModel(world))
    assert p.obligations == ("green_in_box",)
    assert p.unsupported and "stack" in p.unsupported[0]


def test_grounding_repairs_once_then_fails(world):
    class Bad:
        calls = 0

        def complete(self, system, user, images=()):
            Bad.calls += 1
            return '{"steps": [{"id": "s1", "skill": "fly", "object_id": "red"}], "obligations": []}'

    with pytest.raises(ProcedureError, match="grounding failed"):
        ground_goal("put red in the box", world, Bad(), max_repairs=1)
    assert Bad.calls == 2


def test_compiled_procedure_runs_on_replay_with_a_new_goal(world, tmp_path):
    p = ground_goal("put the green block in the left bin and the blue block in the right bin", world,
                    KeywordGroundingModel(world))
    scenario = scenario_with_procedure(load_scenario(WORKBENCH), p)
    scene = ReplayScene(scenario, tmp_path / "scene.json")
    ctx = build_context("configs/robot/mock_dual.yaml", backend="mock",
                        policy_backend=ReplayBackend(scene, lie_first=True))
    observer = ReplayObserver(ctx.world, scene, ttl_s=30.0, bound=False)
    report = asyncio.run(run_authored(scenario, ctx, compile_procedure(p, world), observer))
    assert report.completed
    assert [a.status for a in report.attempts] == ["failed", "verified", "verified"]


@pytest.mark.parametrize("mode", ["planner", "authored"])
def test_service_runs_a_free_text_goal_in_both_modes(tmp_path, mode):
    from liminal.orchestration import service

    opts = service.RunOptions(WORKBENCH, mode=mode, goal="put the blue block in the right bin and the green "
                              "block in the box", db=str(tmp_path / "m.sqlite3") if mode == "planner" else None,
                              log_dir=str(tmp_path), lie_first=True)
    prepared = service.prepare(opts)
    report = asyncio.run(service.execute(prepared))
    assert report.completed, report.summary()
    assert {o.obligation_id for o in report.obligations} == {"blue_in_right_bin", "green_in_box"}
    assert any(e["kind"] == "episode_stop" for e in prepared.ctx.world.events)


def test_service_revalidates_a_client_procedure(tmp_path):
    from liminal.orchestration import service

    forged = {"steps": [{"id": "s1", "skill": "pick_place", "object_id": "red", "destination_id": "box",
                         "effect": "blue_in_box"}], "obligations": ["blue_in_box"]}
    with pytest.raises(ProcedureError):
        service.prepare(service.RunOptions(WORKBENCH, mode="authored", procedure=forged, log_dir=str(tmp_path)))


def test_cli_goal_plan_only_and_frozen_resume(tmp_path, capsys):
    from liminal.ui.cli import main

    goal = "put the blue and green blocks in the box"
    assert main(["episode", WORKBENCH, "--goal", goal, "--plan-only"]) == 0
    assert 'FLUX: "Pick up the green block and place it inside the white box."' in capsys.readouterr().out
    db = str(tmp_path / "m.sqlite3")
    common = ["episode", WORKBENCH, "--db", db, "--log-dir", str(tmp_path)]
    assert main([*common, "--goal", goal, "--interrupt-after", "1"]) == 1
    run_id = next(tmp_path.glob("*.procedure.json")).name.split(".")[2]
    # The resume re-uses the frozen, accepted procedure: no --goal needed, no regrounding.
    assert main([*common, "--resume", run_id]) == 0
    out = capsys.readouterr().out
    assert "completed (completed=True)" in out and "green_in_box: verified" in out


def test_cli_goal_with_only_unsupported_actions_does_not_run(tmp_path):
    from liminal.ui.cli import main

    assert main(["episode", WORKBENCH, "--goal", "stack the blue block on the red block", "--authored",
                 "--log-dir", str(tmp_path)]) == 2


LID = "configs/replay/lid_hold_insert_slip.yaml"


def lid_procedure():
    return {"goal_text": "hold the lid open with the left arm and insert the block with the right arm",
            "steps": [{"id": "open", "skill": "open_lid", "arm": "left", "effect": "lid_open"},
                      {"id": "insert", "skill": "insert", "object_id": "block", "destination_id": "slot",
                       "arm": "right", "maintain": ["lid_open"], "depends_on": ["open"]}],
            "obligations": ["block_in_slot"]}


def test_maintained_step_compiles_to_a_cooperative_hold_and_survives_a_slip(tmp_path):
    from liminal.orchestration import service

    prepared = service.prepare(service.RunOptions(LID, mode="authored", procedure=lid_procedure(),
                                                  log_dir=str(tmp_path)))
    raw = prepared.task.raw["tree"]["sequence"]
    block = raw[0]["retry"]["do"]
    assert block["sequence"][1]["while"]["arm"] == "left" and block["sequence"][1]["while"]["maintain"] == "lid_open"
    assert raw[1] == {"skill": {"arm": "left", "name": "release"}}
    report = asyncio.run(service.execute(prepared))
    assert report.completed, report.summary()
    assert [(a.skill, a.status) for a in report.attempts][:2] == [("open_lid", "verified"), ("insert", "failed")]
    assert any(e["kind"] == "while_violated" for e in prepared.ctx.world.events)


@pytest.mark.parametrize("change, match", [
    (lambda p: p["steps"][1].update(maintain=["lid_closed"]), "earlier step"),
    (lambda p: p["steps"][1].update(arm="left"), "cannot run"),
])
def test_cooperative_compile_rejects_unsupported_shapes(tmp_path, change, match):
    from liminal.orchestration import service

    proc = lid_procedure()
    change(proc)
    with pytest.raises(ProcedureError, match=match):
        service.prepare(service.RunOptions(LID, mode="authored", procedure=proc, log_dir=str(tmp_path)))


def test_planner_mode_honours_the_operator_arm_choice(tmp_path):
    from liminal.orchestration import service

    goal = "put the green block in the left bin, then the red block in the right bin with the right arm"
    prepared = service.prepare(service.RunOptions(WORKBENCH, mode="planner", goal=goal,
                                                  db=str(tmp_path / "m.sqlite3"), log_dir=str(tmp_path)))
    report = asyncio.run(service.execute(prepared))
    assert report.completed
    assert {a.obligation_ids[0]: a.arm for a in report.attempts} == {"green_in_left_bin": "left",
                                                                     "red_in_right_bin": "right"}


def test_validator_rejects_a_planner_that_reassigns_an_operator_arm():
    from liminal.orchestration.plan_validation import PlanningContext, PlanValidationError, validate_proposal
    from liminal.orchestration.replay import ReplayModel

    scenario = load_scenario(WORKBENCH)
    ctx = PlanningContext("run-1", 0, scenario.goal, (), (), dict(scenario.objects), scenario.registry,
                          progress={"completed_effects": [], "operator_arms": {"red_in_box": "right"}})
    import json

    proposal = json.loads(ReplayModel(scenario).complete("", json.dumps({
        "run_id": "run-1", "revision": 0, "trusted_goal": {"id": scenario.goal.goal_id, "text": "x",
                                                           "obligations": ["red_in_box"]},
        "frames": [], "objects": dict(scenario.objects), "available_arms": ["left", "right"],
        "runtime_capabilities": [{"skill_id": "pick_place", "arms": ["left", "right"],
                                  "effect_templates": ["{object_id}_in_{destination_id}"]}],
        "progress": {"completed_effects": []}})))
    assert proposal["steps"][0]["arm"] == "left"
    with pytest.raises(PlanValidationError, match="operator assigned"):
        validate_proposal(proposal, ctx)


def test_offline_grounder_reports_unknown_objects_instead_of_guessing(world):
    p = ground_goal("Put the cable away and the red pencil on the container, the rest go into another container",
                    world, KeywordGroundingModel(world))
    assert not p.steps and any("cable" in u for u in p.unresolved) and any("pencil" in u for u in p.unresolved)


def test_pick_place_without_destination_is_rejected(world):
    with pytest.raises(ProcedureError, match="needs a known destination"):
        parse_procedure({"steps": [step(destination_id="")], "obligations": []}, world)
