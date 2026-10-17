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

    log = Path(args.log_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_{Path(args.task).stem}.jsonl"
    ctx = build_context(args.robot, backend=args.backend, policy=args.policy, log_path=log)
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

    uvicorn.run(create_app(args.robot, args.backend, args.policy, args.task_dir), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="liminal")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="execute a task file")
    r.add_argument("task")
    r.add_argument("--robot", default="configs/robot/mock_dual.yaml")
    r.add_argument("--backend", default="mock", choices=["mock", "scripted", "flux"])
    r.add_argument("--policy", default=None, help="policy config, e.g. configs/policy/flux3_so101.yaml")
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
    s.add_argument("--task-dir", default="configs/tasks")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=_serve)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
