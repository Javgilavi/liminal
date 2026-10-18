"""Box assembly task built with the existing task grammar, in any placement order."""
from __future__ import annotations

from collections.abc import Sequence

from liminal.orchestration.task_spec import TaskSpec, parse_task

COMPONENTS = ("green_pen", "red_pen", "yellow_boxcutter")
DEFINITIONS = {
    "lid_open": "The box lid is open far enough for the left arm to reach every component slot safely.",
    **{f"{item}_in_slot": f"The {item.replace('_', ' ')} is seated in its own slot and released by the gripper."
       for item in COMPONENTS},
    "components_assembled": "The green pen, red pen and yellow boxcutter are all seated in their respective slots, "
                            "with none still held by a gripper.",
}
PROMPTS = {
    "green_pen": "put the green pen in the green slot",
    "yellow_boxcutter": "put the yellow cutter in the yellow slot",
    "red_pen": "put the red pen in its slot in the cardboard box",
}


def assembly_task(order: Sequence[str] = COMPONENTS) -> TaskSpec:
    """Each item depends only on an open lid, never another item's placement."""
    if isinstance(order, str) or len(order) != len(COMPONENTS) or set(order) != set(COMPONENTS):
        raise ValueError(f"placement order must contain each of {COMPONENTS} exactly once")

    def inspect(*predicates):
        return {"inspect": {"predicates": list(predicates), "cameras": ["top", "left_wrist"],
                            "definitions": {key: DEFINITIONS[key] for key in predicates}, "timeout_s": 10}}

    placements = []
    for item in order:
        predicate = f"{item}_in_slot"
        placements.extend([
            inspect("lid_open", predicate),
            {"check": "lid_open"},
            {"fallback": [
                {"check": predicate},
                {"skill": {"arm": "left", "name": f"put_{item}_in_slot", "target": item,
                           "prompt": PROMPTS[item], "success": predicate, "timeout_s": 240,
                           "verify_timeout_s": 2, "zones": ["box"]}},
            ]},
        ])
    predicates = [f"{item}_in_slot" for item in COMPONENTS]
    placements.extend([inspect("lid_open", *predicates, "components_assembled"),
                       *[{"check": key} for key in ("lid_open", *predicates, "components_assembled")]])
    return parse_task({
        "name": "assemble_components",
        "description": "Right helper opens and holds the box; left main arm places three components in any order.",
        "goal": "components_assembled",
        "placement_order": list(order),
        "tree": {"sequence": [
            inspect("lid_open"),
            {"fallback": [
                {"check": "lid_open"},
                {"skill": {"arm": "right", "name": "open_box", "target": "lid",
                           "prompt": "open the box by pulling its handle",
                           "success": "lid_open", "timeout_s": 240, "zones": ["box"]}},
            ]},
            {"while": {"arm": "right", "maintain": "lid_open", "until": "components_assembled",
                       "do": {"sequence": placements}}},
            {"check": "components_assembled"},
        ]},
    }, arms=["left", "right"], skills=["open_box", *[f"put_{item}_in_slot" for item in COMPONENTS]])
