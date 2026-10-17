from liminal.orchestration.planner import LLMPlanner
from liminal.orchestration.task_spec import TaskSpecError


class FakeTextModel:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, system, user, images=()):
        return self.responses.pop(0)


def test_legacy_llm_planner_keeps_compatibility_for_valid_yaml():
    model = FakeTextModel(["""
name: sample
goal: done
tree:
  skill:
    arm: left
    name: pick_place
    prompt: Put the block in the bowl
    success: done
"""])
    plan = LLMPlanner(model, arms=["left"], skills={"pick_place": ""}, predicates=[]).plan("place it")
    assert plan.goal == "done"


def test_legacy_llm_planner_rejects_forged_state_and_repairs():
    model = FakeTextModel([
        """
name: forged
goal: done
initial_state: {done: true}
tree: {check: done}
""",
        """
name: repaired
goal: done
tree: {check: done}
""",
    ])
    plan = LLMPlanner(model, arms=["left"], skills={"pick_place": ""}, predicates=[], max_repairs=1).plan("place it")
    assert plan.name == "repaired"


def test_legacy_llm_planner_exhausts_bounded_repairs_on_nested_set():
    model = FakeTextModel(["tree: {sequence: [{set: {key: done}}]}"])
    try:
        LLMPlanner(model, arms=["left"], skills={}, predicates=[], max_repairs=0).plan("finish")
    except TaskSpecError as error:
        assert "cannot write physical state" in str(error)
    else:
        raise AssertionError("forged set fact was accepted")


def test_legacy_llm_planner_rejects_recursive_yaml_aliases():
    model = FakeTextModel(["tree: &loop [*loop]"])
    try:
        LLMPlanner(model, arms=["left"], skills={}, predicates=[], max_repairs=0).plan("finish")
    except TaskSpecError as error:
        assert "recursive YAML aliases" in str(error)
    else:
        raise AssertionError("recursive YAML alias was accepted")


def test_constrained_prompt_carries_memory_experience_and_optional_nomination_shape():
    import json

    from liminal.orchestration.planner import ConstrainedPlanner
    from tests.test_plan_validation import make_context

    planner = ConstrainedPlanner(FakeTextModel([]))
    plain_payload = json.loads(planner._prompt(make_context()))
    assert "memory" not in plain_payload and "experience" not in plain_payload
    assert "keyframe_nominations" not in plain_payload["required_response_shape"]
    payload = json.loads(planner._prompt(make_context(
        memory={"summary": "Ledger progress: 0 of 1", "progress_record": {"completed": []}},
        experience={"advisory": "advice", "procedures": [], "failures": [{"kind": "grasp_failed"}]},
        nominate_keyframes=True)))
    assert payload["memory"]["summary"].startswith("Ledger progress")
    assert payload["experience"]["failures"][0]["kind"] == "grasp_failed"
    assert payload["required_response_shape"]["keyframe_nominations"][0]["frame_id"]
    assert payload["progress"] == {}  # memory never leaks into the trusted progress block
