import pytest

from liminal.orchestration.task_spec import TaskSpecError, parse_task


@pytest.mark.parametrize("kwargs", [{"arms": []}, {"skills": []}])
def test_empty_allowlist_denies_all(kwargs):
    with pytest.raises(TaskSpecError, match="unknown"):
        parse_task({"tree": {"skill": {"arm": "left", "name": "pick_place"}}}, **kwargs)


@pytest.mark.parametrize("tree", [
    {"retry": {"attempts": True, "do": {"check": "ok"}}},
    {"retry": {"attempts": 1.5, "do": {"check": "ok"}}},
    {"timeout": {"seconds": float("nan"), "do": {"check": "ok"}}},
    {"wait": {"condition": "ok", "timeout_s": -1}},
    {"skill": {"arm": "left", "name": [], "timeout_s": 1}},
    {"skill": {"arm": "left", "name": "x", "zones": "box"}},
    {"retry": {"do": {"check": "ok"}, "invented": True}},
    {"node": {}},
])
def test_bad_types_and_extra_keys_are_actionable(tree):
    with pytest.raises(TaskSpecError, match="tree"):
        parse_task({"tree": tree})


def test_nested_tree_budget():
    node = {"check": "ok"}
    for _ in range(40):
        node = {"sequence": [node]}
    with pytest.raises(TaskSpecError, match="budget"):
        parse_task({"tree": node})
