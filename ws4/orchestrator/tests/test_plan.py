"""Plan format, validation and the offline stub planner (no MuJoCo)."""

import json

import numpy as np
import pytest
from orchestrator.plan import (
    Plan,
    Step,
    concurrent_pairs,
    fill_defaults,
    levels,
    parse_plan,
    plan_json_schema,
    validate_plan,
)
from orchestrator.planner import StubPlanner, arm_by_side

GOAL = "put the apple and the yellow block in the box, then close the lid"
POSITIONS = {"apple": np.array([0.2, 0.16, 0.02]), "yellow_block": np.array([0.2, -0.16, 0.02]),
             "lid": np.array([0.08, -0.3, 0.0])}


def stub(vocab):
    return StubPlanner(vocab, arm_by_side(lambda: POSITIONS))


def test_flux_prompts_match_the_pod_prewarm_list(vocab):
    """FLUX conditions on the exact caption; the pod pre-encodes ws4/cloud/warm_prompts.py's strings."""
    import importlib.util

    from orchestrator import WS4

    spec = importlib.util.spec_from_file_location("warm_prompts", WS4 / "cloud" / "warm_prompts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(vocab.flux_prompts()) <= set(module.DEFAULT_PROMPTS)


def test_stub_plan_for_the_demo_goal_is_valid_and_concurrent(vocab):
    result = stub(vocab).plan(GOAL)
    assert result.ok, result.errors
    plan = result.plan
    by = {s.skill: s for s in plan.steps}
    assert by["put_apple_in_box"].arm == "left" and by["put_block_in_box"].arm == "right"
    assert by["close_lid"].arm == "right"
    assert set(by["close_lid"].depends_on) == {by["put_apple_in_box"].id, by["put_block_in_box"].id}
    assert (by["put_apple_in_box"].id, by["put_block_in_box"].id) in concurrent_pairs(plan)
    assert all(s.success_check for s in plan.steps)


def test_stub_plan_with_open_and_hold(vocab):
    result = stub(vocab).plan("open the lid, then put the apple and the yellow block in the box, then close the lid")
    assert result.ok, result.errors
    plan = result.plan
    holds = [s for s in plan.steps if s.skill == "hold"]
    assert len(holds) == 1 and holds[0].arm == "left"
    held_for = {plan.step(u).skill for u in holds[0].hold_until}
    assert held_for == {"put_apple_in_box", "put_block_in_box"}
    assert all(plan.step(u).arm == "right" for u in holds[0].hold_until)
    close = next(s for s in plan.steps if s.skill == "close_lid")
    assert holds[0].id in close.depends_on or close.arm == "right"
    assert any(holds[0].id in pair for pair in concurrent_pairs(plan))


def test_stub_rejects_unknown_goal(vocab):
    assert not StubPlanner(vocab).plan("make me a sandwich").ok


def test_stub_replan_skips_finished_skills(vocab):
    state = {"done": [{"id": "put_apple_in_box", "arm": "left", "skill": "put_apple_in_box"}]}
    result = stub(vocab).plan(GOAL, state=state)
    assert result.ok, result.errors
    assert "put_apple_in_box" not in [s.skill for s in result.plan.steps]


def _plan(*steps):
    return Plan([Step(**s) for s in steps])


GOOD = [
    dict(id="a", arm="left", skill="put_apple_in_box", success_check="x", max_attempts=2),
    dict(id="b", arm="right", skill="put_block_in_box", success_check="x", max_attempts=2),
    dict(id="c", arm="right", skill="close_lid", depends_on=["a", "b"], success_check="x", max_attempts=2),
]


def test_good_plan_validates(vocab):
    plan = _plan(*GOOD)
    assert validate_plan(plan, vocab) == []
    assert levels(plan) == {"a": 0, "b": 0, "c": 1}
    assert concurrent_pairs(plan) == [("a", "b")]


@pytest.mark.parametrize("mutate, expect", [
    (lambda s: s[0].update(skill="dance"), "not in the vocabulary"),
    (lambda s: s[0].update(skill="open_lid", arm="right"), "can only run on"),
    (lambda s: s[2].update(depends_on=["a", "zz"]), "unknown step 'zz'"),
    (lambda s: s[0].update(depends_on=["c"]), "dependency cycle"),
    (lambda s: s[2].update(depends_on=["a"]), "neither depends on the other"),
    (lambda s: s[0].update(max_attempts=0), "max_attempts"),
    (lambda s: s[0].update(max_attempts=5), "max_attempts"),
    (lambda s: s[1].update(id="a"), "used twice"),
    (lambda s: s[0].update(hold_until=["b"]), "only hold steps"),
    (lambda s: s[0].update(success_check=""), "success_check"),
    (lambda s: s[0].update(id="has space"), "letters, digits"),
])
def test_validation_errors(vocab, mutate, expect):
    steps = [dict(s) for s in GOOD]
    mutate(steps)
    errors = validate_plan(_plan(*steps), vocab)
    assert any(expect in e for e in errors), errors


def test_hold_rules(vocab):
    base = [
        dict(id="open", arm="left", skill="open_lid", success_check="x", max_attempts=1),
        dict(id="hold", arm="left", skill="hold", depends_on=["open"], hold_until=["apple"], max_attempts=1),
        dict(id="apple", arm="right", skill="put_apple_in_box", depends_on=["open"], success_check="x",
             max_attempts=1),
        dict(id="close", arm="left", skill="close_lid", depends_on=["hold", "apple"], success_check="x",
             max_attempts=1),
    ]
    assert validate_plan(_plan(*base), vocab) == []
    same_arm = [dict(s) for s in base]
    same_arm[1]["hold_until"] = ["close"]
    assert any("same arm" in e for e in validate_plan(_plan(*same_arm), vocab))
    deadlock = [dict(s) for s in base]
    deadlock[2]["depends_on"] = ["close"]  # apple waits for close, close waits for the hold, the hold waits for apple
    deadlock[3]["depends_on"] = ["hold"]
    errors = validate_plan(_plan(*deadlock), vocab)
    assert any("deadlock" in e or "cycle" in e for e in errors), errors
    early = [dict(s) for s in base]
    early[1]["hold_until"] = ["apple"]
    early[1]["depends_on"] = ["open", "apple"]
    assert any("would end at once" in e for e in validate_plan(_plan(*early), vocab))
    empty = [dict(s) for s in base]
    empty[1]["hold_until"] = []
    assert any("must list" in e for e in validate_plan(_plan(*empty), vocab))


def test_replan_may_depend_on_finished_steps(vocab):
    plan = _plan(dict(id="c2", arm="right", skill="close_lid", depends_on=["a"], success_check="x", max_attempts=1))
    assert validate_plan(plan, vocab, done={"a"}) == []
    assert any("unknown step" in e for e in validate_plan(plan, vocab))
    reused = _plan(dict(id="a", arm="right", skill="close_lid", success_check="x", max_attempts=1))
    assert any("already used" in e for e in validate_plan(reused, vocab, done={"a"}))


def test_parse_plan_schema_errors():
    assert parse_plan("{not json")[1][0].startswith("not valid JSON")
    assert parse_plan({"rationale": ""})[1] == ["'steps' must be a list"]
    plan, errors = parse_plan({"steps": [{"id": 3, "arm": "left", "skill": "x", "depends_on": "a",
                                          "max_attempts": "2", "colour": "red"}]})
    assert plan is None
    text = " ".join(errors)
    for needle in ("unknown keys", "id must be", "depends_on must be", "max_attempts must be"):
        assert needle in text


def test_round_trip_and_defaults(vocab):
    raw = {"rationale": "r", "steps": [dict(s, success_check="", depends_on=s.get("depends_on", []),
                                            hold_until=[]) for s in GOOD]}
    plan, errors = parse_plan(json.dumps(raw), goal="g")
    assert errors == []
    fill_defaults(plan, vocab)
    assert plan.step("a").success_check == vocab["put_apple_in_box"].success_check
    again, _ = parse_plan(plan.to_json())
    assert [s.id for s in again.steps] == ["a", "b", "c"]


def test_json_schema_is_closed_for_structured_outputs(vocab):
    schema = plan_json_schema(vocab)

    def walk(node):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert sorted(node["required"]) == sorted(node["properties"])
            for child in node["properties"].values():
                walk(child)
        if node.get("type") == "array":
            walk(node["items"])

    walk(schema)
    assert schema["properties"]["steps"]["items"]["properties"]["skill"]["enum"] == vocab.names()
