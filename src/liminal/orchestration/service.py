"""One entry point for building and running harness episodes (shared by the CLI and the web UI).

Everything here runs offline by default: mock arms, the replay scene as the "world", the replay
observer as the camera, and scripted models. ``model="claude"`` swaps in the real planner/grounder.
Hardware runs are not wired through this module yet (WS1 FLUX loader, WS5 Haller transport).
"""

from __future__ import annotations

import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from liminal.orchestration.procedure import (
    KeywordGroundingModel,
    ProcedureProposal,
    SkillWorld,
    compile_procedure,
    describe,
    ground_goal,
    parse_procedure,
    scenario_with_procedure,
)

SCENARIO_DIR = Path("configs/replay")
TASK_DIR = Path("configs/tasks")
DEMO_DIR = Path("configs/demos")
MODELS = ("replay", "claude")
MODES = ("planner", "authored")
DEFAULT_EXPERIENCE_DB = "runs/experience.sqlite3"
# run_episode memory kwargs the UI/CLI offer (the first choice is run_episode's default).
MEMORY_OPTIONS: dict[str, dict[str, Any]] = {
    "keyframes": {"type": "choice", "choices": ["event", "nominated", "none"], "default": "event"},
    "max_keyframes": {"type": "number", "default": 4},
    "experience_mode": {"type": "choice", "choices": ["off", "read", "readwrite"], "default": "off"},
    "experience_frozen": {"type": "boolean", "default": False},
}


@dataclass
class RunOptions:
    scenario: str
    mode: str = "planner"  # planner: one validated step at a time (V2/V3); authored: compiled tree (V1)
    model: str = "replay"
    goal: str | None = None  # free text, grounded before the run unless `procedure` is given
    procedure: dict[str, Any] | None = None  # an operator-accepted ProcedureProposal.to_dict(); re-validated
    task: str | None = None  # authored task file (mode=authored without a procedure)
    db: str | None = None
    run_id: str | None = None
    resume: bool = False
    interrupt_after: int | None = None
    fail_first: bool = False
    lie_first: bool = False
    max_attempts: int = 6
    max_model_calls: int | None = None
    robot: str = "configs/robot/mock_dual.yaml"
    log_dir: str = "runs"
    scene_state: str | None = None
    memory: dict[str, Any] = field(default_factory=dict)  # extra run_episode memory options (V3+/V6)


@dataclass
class Prepared:
    """A built, not-yet-started run. `ctx.world.events` is the live trace (the UI streams it)."""

    options: RunOptions
    run_id: str
    scenario: Any
    scene: Any
    ctx: Any
    observer: Any
    log_path: Path
    procedure: ProcedureProposal | None = None
    task: Any = None


def list_options() -> dict[str, Any]:
    def names(d: Path, pattern: str = "*.yaml") -> list[str]:
        return sorted(str(p) for p in d.glob(pattern)) if d.exists() else []

    return {"scenarios": names(SCENARIO_DIR), "tasks": names(TASK_DIR, "replay_*.yaml"),
            "demos": sorted(str(p) for p in DEMO_DIR.iterdir() if p.is_dir()) if DEMO_DIR.exists() else [],
            "models": list(MODELS), "modes": list(MODES), "memory": MEMORY_OPTIONS}


def skill_world(scenario: Any) -> SkillWorld:
    return SkillWorld(scenario.objects, scenario.registry)


def text_model(kind: str, *, purpose: str = "grounding", world: SkillWorld | None = None) -> Any:
    """`replay` -> deterministic offline model; `claude` -> the configured Claude model (needs .env key)."""
    if kind == "replay":
        if purpose != "grounding" or world is None:
            raise ValueError("the offline replay model only grounds goals here")
        return KeywordGroundingModel(world)
    if kind == "claude":
        from liminal.orchestration.episode import make_model

        return make_model("claude")
    raise ValueError(f"unknown model {kind!r} (expected one of {MODELS})")


def ground(scenario_path: str, goal: str, model: str = "replay") -> tuple[ProcedureProposal, list[dict[str, Any]]]:
    """Free text -> validated procedure + the atomic-skill breakdown to show the operator."""
    from liminal.orchestration.replay import load_scenario

    scenario = load_scenario(scenario_path)
    world = skill_world(scenario)
    proposal = ground_goal(goal, world, text_model(model, world=world))
    return proposal, describe(proposal, world)


def ground_demo(scenario_path: str, demo_path: str, model: str = "replay", *, strategy: str = "sampled",
                max_frames: int = 8) -> tuple[ProcedureProposal, list[dict[str, Any]], dict[str, Any]]:
    """Demonstration video/frames -> the same validated procedure + rows, plus the stage-1 extraction."""
    from liminal.demo import ground_demo_path

    return ground_demo_path(scenario_path, demo_path, model, strategy=strategy, max_frames=max_frames)


def memory_kwargs(memory: dict[str, Any]) -> dict[str, Any]:
    """UI/CLI memory options -> run_episode kwargs (experience needs a db; default a shared one)."""
    unknown = set(memory) - set(MEMORY_OPTIONS) - {"experience_db", "keyframe_gap"}
    if unknown:
        raise ValueError(f"unknown memory options {sorted(unknown)}")
    kwargs = dict(memory)
    if kwargs.get("experience_mode", "off") != "off":
        kwargs.setdefault("experience_db", DEFAULT_EXPERIENCE_DB)
        Path(kwargs["experience_db"]).parent.mkdir(parents=True, exist_ok=True)
    return kwargs


def prepare(options: RunOptions) -> Prepared:
    from liminal.app import build_context
    from liminal.orchestration import load_task
    from liminal.orchestration.replay import (
        ReplayBackend,
        ReplayObserver,
        ReplayScene,
        attach_replay_primitives,
        load_scenario,
    )

    if options.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if options.mode == "planner" and not options.db:
        raise ValueError("planner runs need a memory db path")
    if options.mode == "authored" and (options.resume or options.interrupt_after is not None):
        raise ValueError("authored runs have no ledger: resume/interrupt need the planner mode")
    scenario = load_scenario(options.scenario)
    world = skill_world(scenario)
    procedure = None
    if options.procedure is not None:
        procedure = parse_procedure(options.procedure, world)  # never trust the client copy
    elif options.goal:
        procedure = ground_goal(options.goal, world, text_model(options.model, world=world))
    if procedure is not None:
        scenario = scenario_with_procedure(scenario, procedure)
    run_id = options.run_id or f"run-{uuid.uuid4().hex[:8]}"
    if options.db:
        state_path = options.scene_state or f"{options.db}.{run_id}.scene.json"
    else:
        state_path = options.scene_state or str(Path(tempfile.mkdtemp(prefix="liminal-scene-")) / "scene.json")
    scene = ReplayScene(scenario, state_path)
    log = Path(options.log_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_episode_{run_id}.jsonl"
    ctx = build_context(options.robot, backend="mock", log_path=log,
                        policy_backend=ReplayBackend(scene, fail_first=options.fail_first,
                                                     lie_first=options.lie_first))
    attach_replay_primitives(ctx, scene)  # releasing a support arm is visible to the replay scene
    observer = ReplayObserver(ctx.world, scene, ttl_s=30.0, bound=options.mode == "planner")
    task = None
    if options.mode == "authored":
        if procedure is not None:
            task = compile_procedure(procedure, world, name=f"procedure_{scenario.name}")
        elif options.task:
            task = load_task(options.task, arms=ctx.arm_names)
        else:
            raise ValueError("authored mode needs a goal, a procedure or a task file")
    return Prepared(options, run_id, scenario, scene, ctx, observer, log, procedure, task)


async def execute(p: Prepared, on_start: Callable[[Prepared], None] | None = None) -> Any:
    """Run a prepared episode to its stop reason; returns the EpisodeReport."""
    from liminal import settings
    from liminal.orchestration.episode import make_model, run_authored, run_episode

    o = p.options
    if on_start is not None:
        on_start(p)
    for arm in p.ctx.arms.values():
        arm.connect()
    try:
        if o.mode == "authored":
            return await run_authored(p.scenario, p.ctx, p.task, p.observer, run_id=p.run_id)
        model = make_model(o.model, p.scenario, p.scene)
        max_calls = o.max_model_calls
        if max_calls is None:
            settings.load_env_file()
            max_calls = settings.positive_setting("LIMINAL_MAX_MODEL_CALLS", 3) if o.model == "claude" else 20
        return await run_episode(
            p.scenario, p.ctx, model, o.db, p.observer.observe, observer_ids=(p.observer.OBSERVER_ID,),
            run_id=p.run_id, max_attempts=o.max_attempts, max_model_calls=max_calls,
            interrupt_after=o.interrupt_after, mode="replay", **memory_kwargs(o.memory),
            arm_hints={s.effect: s.arm for s in p.procedure.steps if s.arm} if p.procedure else None)
    finally:
        for arm in p.ctx.arms.values():
            arm.disconnect()
