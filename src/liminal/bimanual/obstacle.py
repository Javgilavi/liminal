"""Bimanual "clear the obstacle, then reach the target" planning (owned by WS2).

Baseline heuristic to iterate on (see docs/research/02_bimanual_obstacle.md):

1. assign roles: the arm closest to the obstacle *clears*, the other *acts* on the
   target (ties/unreachable -> the single reachable arm does both, sequentially)
2. pick a clearing strategy per obstacle:
     push_aside - slide it out of the corridor (no grasp; robust, low precision)
     pick_aside - grasp and drop it in a free zone
     hold_away  - lift/open it and HOLD while the other arm works (lids, flaps,
                  anything that falls back) -> emitted as a `while` node
3. emit a task tree in the task_spec grammar, so it runs on the same executor

Positions are table-plane (x, y) in metres in a shared frame (calibrate once, see
docs/architecture.md). Reachability is a disc around each arm base for now.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

Strategy = Literal["push_aside", "pick_aside", "hold_away"]


@dataclass
class ArmBase:
    name: str
    xy: tuple[float, float]
    reach_m: float = 0.30

    def reaches(self, p: tuple[float, float]) -> bool:
        return math.dist(self.xy, p) <= self.reach_m


@dataclass
class SceneObject:
    name: str
    xy: tuple[float, float]
    strategy: Strategy = "push_aside"
    blocks: list[str] = field(default_factory=list)  # names of objects it occludes


@dataclass
class Scene:
    arms: list[ArmBase]
    objects: list[SceneObject]

    def obj(self, name: str) -> SceneObject:
        return next(o for o in self.objects if o.name == name)


class NoFeasiblePlan(RuntimeError):
    pass


def blockers_of(scene: Scene, target: str) -> list[SceneObject]:
    return [o for o in scene.objects if target in o.blocks]


def assign_roles(scene: Scene, obstacle: SceneObject, target: SceneObject) -> tuple[ArmBase, ArmBase]:
    """Return (clearing_arm, acting_arm). Prefers two different arms."""
    can_clear = [a for a in scene.arms if a.reaches(obstacle.xy)]
    can_act = [a for a in scene.arms if a.reaches(target.xy)]
    if not can_clear or not can_act:
        raise NoFeasiblePlan(f"no arm reaches {'obstacle' if not can_clear else 'target'}")
    best: tuple[float, ArmBase, ArmBase] | None = None
    for c in can_clear:
        for a in can_act:
            if c.name == a.name and obstacle.strategy == "hold_away":
                continue  # one arm cannot hold and act at once
            cost = math.dist(c.xy, obstacle.xy) + math.dist(a.xy, target.xy) + (0.5 if c.name == a.name else 0.0)
            if best is None or cost < best[0]:
                best = (cost, c, a)
    if best is None:
        raise NoFeasiblePlan(f"{obstacle.name} needs holding but only one arm reaches both")
    return best[1], best[2]


def plan_clear_and_act(scene: Scene, target: str, action_prompt: str, goal: str,
                       action_skill: str = "pick_place") -> dict[str, Any]:
    """Build a task spec dict: clear every blocker of `target`, then act on it."""
    tgt = scene.obj(target)
    steps: list[dict[str, Any]] = []
    act_arm: str | None = None
    held: list[str] = []
    act_node = {"retry": {"attempts": 2, "do": {"skill": {
        "arm": "?", "name": action_skill, "target": target, "prompt": action_prompt, "success": goal,
        "timeout_s": 30}}}}

    holds: list[tuple[str, str]] = []  # (arm, predicate) to wrap around the action
    for obs in blockers_of(scene, target):
        clear_arm, acting = assign_roles(scene, obs, tgt)
        act_arm = acting.name
        cleared = f"{obs.name}_cleared"
        if obs.strategy == "hold_away":
            steps.append({"skill": {"arm": clear_arm.name, "name": "lift_aside", "target": obs.name,
                                    "prompt": f"Lift the {obs.name} up and out of the way", "success": cleared}})
            holds.append((clear_arm.name, cleared))
            held.append(clear_arm.name)
        else:
            steps.append({"retry": {"attempts": 2, "do": {"skill": {
                "arm": clear_arm.name, "name": obs.strategy, "target": obs.name,
                "prompt": (f"Push the {obs.name} to the side" if obs.strategy == "push_aside"
                           else f"Pick up the {obs.name} and put it in the free zone"),
                "success": cleared}}}})

    if act_arm is None:  # nothing in the way
        act_arm = min((a for a in scene.arms if a.reaches(tgt.xy)), key=lambda a: math.dist(a.xy, tgt.xy)).name
    act_node["retry"]["do"]["skill"]["arm"] = act_arm

    node: dict[str, Any] = act_node
    for arm, pred in reversed(holds):
        node = {"while": {"arm": arm, "maintain": pred, "until": goal, "do": node}}
    steps.append(node)
    steps.append({"parallel": [{"skill": {"arm": a.name, "name": "release" if a.name in held else "home"}}
                               for a in scene.arms]})
    steps.append({"check": goal})
    return {"name": f"clear_and_{action_skill}_{target}", "goal": goal, "tree": {"sequence": steps}}
