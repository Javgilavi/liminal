"""Minimal sequential skill harness using Liminal's verified behavior-tree runtime."""

from __future__ import annotations

SKILLS = (
    ("pick_gear", "gear_held", ()),
    ("seat_gear", "gear_seated", ("gear_held",)),
    ("pick_collar", "collar_held", ("gear_seated",)),
    ("seat_collar", "assembly_complete", ("gear_seated", "collar_held")),
)
PROMPTS = {
    "pick_gear": "Pick up the blue gear and hold it upright above the tray.",
    "seat_gear": "Lower the blue gear over the metal shaft, release it onto the base, and move the gripper clear.",
    "pick_collar": "Pick up the brass spacer collar and hold it upright above the tray.",
    "seat_collar": "Lower the brass spacer collar onto the shaft above the blue gear, release it, and move clear.",
}


def assembly_task(arm: str = "sim", attempts: int = 1):
    from liminal.orchestration.task_spec import parse_task

    if attempts < 1:
        raise ValueError("attempts must be positive")
    nodes = []
    for name, success, preconditions in SKILLS:
        body = {
            "sequence": [
                *[{"check": condition} for condition in preconditions],
                {
                    "skill": {
                        "arm": arm,
                        "name": name,
                        "prompt": PROMPTS[name],
                        "success": success,
                        "timeout_s": 180,
                        "verify_timeout_s": 0.01,
                    }
                },
            ]
        }
        nodes.append({"retry": {"attempts": attempts, "do": body}})
    nodes.append({"check": "assembly_complete"})
    return parse_task(
        {"name": "nist_gear_collar_assembly", "goal": "assembly_complete", "tree": {"sequence": nodes}},
        arms=[arm],
        skills=[row[0] for row in SKILLS],
    )
