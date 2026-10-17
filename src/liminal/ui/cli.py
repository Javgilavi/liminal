"""Command-line trigger (owned by WS5).

    liminal run configs/tasks/lid_hold_insert.yaml --robot configs/robot/mock_dual.yaml
    liminal plan "put the block in the box" --planner template
    liminal validate configs/tasks/*.yaml
    liminal serve --robot configs/robot/so101_dual.yaml --backend flux
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import yaml


def _run(args: argparse.Namespace) -> int:
    from liminal.app import build_context
    from liminal.hardware.runtime import connected
    from liminal.orchestration import load_task, run_task
    from liminal.perception.config import load_perception_config
    from liminal.perception.vlm import warm_vision_model

    log = Path(args.log_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_{Path(args.task).stem}.jsonl"
    perception = load_perception_config(args.perception) if args.perception else None
    if perception:
        warm_vision_model(perception.model)
    ctx = build_context(
        args.robot,
        backend=args.backend,
        policy=args.policy,
        log_path=log,
        vision_model=perception.model if perception else None,
        inspection=perception.inspection if perception else None,
        inspection_confirmations=perception.confirmations if perception else 2,
    )
    task = load_task(args.task, arms=ctx.arm_names)
    with connected(ctx):
        try:
            if args.engage:
                for name, arm in ctx.arms.items():
                    if hasattr(arm, "set_torque"):
                        ctx.safety.set_torque(name, True)
            report = asyncio.run(run_task(task, ctx))
        except KeyboardInterrupt:
            ctx.safety.estop()
            print("e-stop: interrupted", file=sys.stderr)
            return 130
    print(json.dumps(report.__dict__, default=str, indent=2))
    print(f"event log: {log}")
    return 0 if report.goal_verified else 1


def _validate(args: argparse.Namespace) -> int:
    from liminal.orchestration import TaskSpecError, load_task

    bad = 0
    for p in args.tasks:
        try:
            load_task(p)
            print(f"ok   {p}")
        except TaskSpecError as e:
            bad += 1
            print(f"FAIL {p}: {e}")
    return 1 if bad else 0


def _plan(args: argparse.Namespace) -> int:
    from liminal.orchestration.planner import ClaudeModel, GeminiModel, LLMPlanner, TemplatePlanner

    if args.planner == "template":
        spec = TemplatePlanner(args.task_dir).plan(args.goal)
        print(yaml.safe_dump(spec.raw, sort_keys=False))
        return 0
    catalog = yaml.safe_load(Path(args.skills).read_text())
    model = ClaudeModel(args.model) if args.planner == "claude" else GeminiModel(args.model)
    planner = LLMPlanner(model, arms=catalog["arms"], skills=catalog["skills"], predicates=catalog["predicates"])
    spec = planner.plan(args.goal)
    print(yaml.safe_dump(spec.raw, sort_keys=False))
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from liminal.ui.server import create_app

    uvicorn.run(
        create_app(args.robot, args.backend, args.policy, args.task_dir, args.perception),
        host=args.host,
        port=args.port,
    )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from PIL import Image

    from liminal.perception.config import load_perception_config
    from liminal.perception.inspector import SceneInspector
    from liminal.perception.qwen import QwenVisionModel
    from liminal.perception.vlm import warm_vision_model
    from liminal.perception.world_state import WorldState

    runtime = load_perception_config(args.perception)
    warm_vision_model(runtime.model)
    image_paths: dict[str, Path] = {}
    for item in args.image:
        camera, separator, value = item.partition("=")
        if not separator or not camera or not value:
            raise SystemExit(f"invalid --image {item!r}; expected CAMERA=PATH")
        if camera in image_paths:
            raise SystemExit(f"duplicate image for camera {camera!r}")
        path = Path(value)
        if not path.is_file():
            raise SystemExit(f"image does not exist: {path}")
        image_paths[camera] = path
    expected_cameras = set(runtime.inspection.cameras)
    if set(image_paths) != expected_cameras:
        raise SystemExit(
            f"images must contain exactly the configured cameras {sorted(expected_cameras)}; "
            f"got {sorted(image_paths)}"
        )

    class ImageCamera:
        def __init__(self, name: str, path: Path):
            self.name, self.path = name, path

        def read(self):
            with Image.open(self.path) as image:
                return image.convert("RGB")

    predicates = tuple(args.predicate or runtime.inspection.definitions)
    request = runtime.inspection.request(predicates, context="Offline setup-photo inspection.")
    world = WorldState()
    inspector = SceneInspector(
        runtime.model,
        {name: ImageCamera(name, path) for name, path in image_paths.items()},
        world,
        confirmations=runtime.confirmations,
    )
    assessment = asyncio.run(inspector.inspect_until_known(
        request,
        timeout_s=runtime.inspection.timeout_s,
        retry_delay_s=runtime.inspection.retry_delay_s,
    ))
    output: dict[str, object] = {
        "predicates": {
            key: {
                "value": observation.value.value,
                "evidence": observation.evidence,
            }
            for key, observation in assessment.observations.items()
        },
        "parse_error": assessment.parse_error,
        "raw_response": assessment.raw_response,
    }
    if isinstance(runtime.model, QwenVisionModel):
        output["load_metrics"] = asdict(runtime.model.load_metrics) if runtime.model.load_metrics else None
        output["inference_metrics"] = [asdict(metrics) for metrics in runtime.model.inference_metrics]
    print(json.dumps(output, indent=2))
    return 0 if all(observation.known for observation in assessment.observations.values()) else 1


def _evaluate_perception(args: argparse.Namespace) -> int:
    from dataclasses import asdict, replace

    from liminal.perception.config import load_perception_config
    from liminal.perception.qwen import QwenVisionModel
    from liminal.perception.replay import evaluate_recording, load_recording
    from liminal.perception.vlm import warm_vision_model

    recording = load_recording(args.labels)
    runtime = load_perception_config(args.perception)
    missing_predicates = set(recording.request.predicates) - set(runtime.inspection.definitions)
    if missing_predicates:
        raise SystemExit(
            f"evaluation predicates are absent from the perception config: {sorted(missing_predicates)}"
        )
    configured_cameras = set(runtime.inspection.cameras)
    if set(recording.request.cameras) != configured_cameras:
        raise SystemExit(
            f"evaluation cameras must exactly match the perception config: {sorted(configured_cameras)}"
        )
    # Labels own expected verdicts and image paths. Prompt semantics stay owned by
    # the deployed perception profile so evaluation cannot silently test a drifted
    # predicate definition or timeout.
    request = runtime.inspection.request(
        recording.request.predicates,
        context=recording.request.context,
    )
    recording = replace(recording, request=request)

    warm_vision_model(runtime.model)
    report = asyncio.run(evaluate_recording(recording, runtime.model, repeats=args.repeats))
    output: dict[str, object] = report.to_dict()
    if isinstance(runtime.model, QwenVisionModel):
        output["model_metrics"] = {
            "config": asdict(runtime.model.config),
            "load": asdict(runtime.model.load_metrics) if runtime.model.load_metrics else None,
            "inferences": [asdict(metrics) for metrics in runtime.model.inference_metrics],
        }

    output_path = Path(args.output) if args.output else (
        Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}_{Path(args.labels).stem}_perception_eval.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    summary = output["summary"]
    assert isinstance(summary, dict)
    print(json.dumps(summary, indent=2))
    print(f"evaluation report: {output_path}")
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="liminal")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="execute a task file")
    r.add_argument("task")
    r.add_argument("--robot", default="configs/robot/mock_dual.yaml")
    r.add_argument("--backend", default="mock", choices=["mock", "scripted", "flux"])
    r.add_argument("--policy", default=None, help="policy config, e.g. configs/policy/flux3_so101.yaml")
    r.add_argument("--perception", default=None, help="persistent VLM + predicate config")
    r.add_argument("--log-dir", default="runs")
    r.add_argument("--engage", action="store_true", help="explicitly engage real arms at their current pose")
    r.set_defaults(fn=_run)

    v = sub.add_parser("validate", help="compile task files without running them")
    v.add_argument("tasks", nargs="+")
    v.set_defaults(fn=_validate)

    p = sub.add_parser("plan", help="goal -> task tree")
    p.add_argument("goal")
    p.add_argument("--planner", default="template", choices=["template", "claude", "gemini"])
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--skills", default="configs/skills.yaml")
    p.add_argument("--task-dir", default="configs/tasks")
    p.set_defaults(fn=_plan)

    s = sub.add_parser("serve", help="HTTP trigger + state API for the operator UI")
    s.add_argument("--robot", default="configs/robot/mock_dual.yaml")
    s.add_argument("--backend", default="mock", choices=["mock", "scripted", "flux"])
    s.add_argument("--policy", default=None)
    s.add_argument("--perception", default=None)
    s.add_argument("--task-dir", default="configs/tasks")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=_serve)

    i = sub.add_parser("inspect", help="run the configured VLM on recorded setup photos")
    i.add_argument("--perception", required=True, help="persistent VLM + predicate config")
    i.add_argument("--image", action="append", required=True, help="camera frame as CAMERA=PATH")
    i.add_argument("--predicate", action="append", help="predicate to inspect; defaults to all definitions")
    i.set_defaults(fn=_inspect)

    e = sub.add_parser("evaluate-perception", help="score the configured VLM on labeled recorded views")
    e.add_argument("labels", help="evaluation YAML manifest")
    e.add_argument("--perception", required=True, help="persistent VLM + predicate config")
    e.add_argument("--repeats", type=int, default=3, help="independent calls per labeled frame (default: 3)")
    e.add_argument("--output", help="JSON report path; defaults to a timestamped file under runs/")
    e.set_defaults(fn=_evaluate_perception)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
