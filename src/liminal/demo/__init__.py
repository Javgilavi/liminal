"""V4: demonstration video -> the same ``ProcedureProposal`` text goals produce. See ``extract`` for the pipeline."""

from __future__ import annotations

from typing import Any

from liminal.demo.corrections import CorrectedProcedure, apply_corrections
from liminal.demo.extract import (
    Binding,
    DemoExtraction,
    bind_role,
    describe_demo,
    ground_demo,
    parse_description,
)
from liminal.demo.replay_model import ReplayDemoModel
from liminal.demo.video import Demo, DemoError, DemoFrame, load_demo

__all__ = ["Binding", "CorrectedProcedure", "Demo", "DemoError", "DemoExtraction", "DemoFrame", "ReplayDemoModel",
           "apply_corrections", "bind_role", "demo_model", "describe_demo", "ground_demo", "ground_demo_path",
           "load_demo", "parse_description"]


def demo_model(kind: str, demo: Demo) -> Any:
    """`replay` -> scripted model reading annotation.yaml; `claude` -> the configured Claude model."""
    if kind == "replay":
        return ReplayDemoModel(demo)
    if kind == "claude":
        from liminal.orchestration.episode import make_model

        return make_model("claude")
    raise ValueError(f"unknown model {kind!r} (expected replay or claude)")


def ground_demo_path(scenario_path: str, demo_path: str, model: str = "replay", *, strategy: str = "sampled",
                     max_frames: int = 8) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Demo folder + replay scenario -> (proposal, atomic-skill rows, extraction dict) for the CLI/UI."""
    from liminal.orchestration.procedure import SkillWorld, describe
    from liminal.orchestration.replay import load_scenario

    scenario = load_scenario(scenario_path)
    world = SkillWorld(scenario.objects, scenario.registry)
    demo = load_demo(demo_path, max_frames=max_frames, strategy=strategy)
    proposal, extraction = ground_demo(demo, world, demo_model(model, demo))
    return proposal, describe(proposal, world), extraction.to_dict()
