"""Command-line trigger (owned by WS5).

    liminal run configs/tasks/lid_hold_insert.yaml --robot configs/robot/mock_dual.yaml
    liminal plan "put the block in the box" --planner template
    liminal validate configs/tasks/*.yaml
    liminal serve --robot configs/robot/so101_dual.yaml --backend flux
    liminal episode configs/replay/two_blocks.yaml --db runs/x.sqlite3 [--interrupt-after 1 | --resume RUN_ID]
    liminal episode configs/replay/two_blocks.yaml --task configs/tasks/replay_two_blocks_v1.yaml
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
    from liminal.orchestration import load_task, run_task

    log = Path(args.log_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_{Path(args.task).stem}.jsonl"
    ctx = build_context(args.robot, backend=args.backend, policy=args.policy, log_path=log)
    task = load_task(args.task, arms=ctx.arm_names)
    for a in ctx.arms.values():
        a.connect()
    try:
        report = asyncio.run(run_task(task, ctx))
    except KeyboardInterrupt:
        ctx.safety.estop()
        print("e-stop: interrupted", file=sys.stderr)
        return 130
    finally:
        for a in ctx.arms.values():
            a.disconnect()
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


def _print_breakdown(proposal, rows) -> None:
    print(f"goal: {proposal.goal_text}")
    print(f"atomic skills ({len(rows)}), obligations: {', '.join(proposal.obligations) or '-'}")
    for r in rows:
        after = f"  after {','.join(r['depends_on'])}" if r["depends_on"] else ""
        print(f"  {r['id']:>4}  {r['arm'].upper():<5} {r['skill']}({r['object_id']} -> {r['destination_id']})"
              f"  FLUX: \"{r['flux_prompt']}\"{after}")
    for item in proposal.unsupported:
        print(f"  UNSUPPORTED (not run): {item}")
    for item in proposal.unresolved:
        print(f"  UNRESOLVED: {item}")


def _memory_options(args: argparse.Namespace) -> dict:
    memory: dict = {}
    if args.keyframes != "event":
        memory["keyframes"] = args.keyframes
    if args.max_keyframes is not None:
        memory["max_keyframes"] = args.max_keyframes
    if args.experience != "off":
        memory["experience_mode"] = args.experience
    if args.experience_db:
        memory["experience_db"] = args.experience_db
    if args.experience_frozen:
        memory["experience_frozen"] = True
    return memory


def _eval(args: argparse.Namespace) -> int:
    from liminal.eval import eval_main

    return eval_main([args.experiment, *(["--out", args.out] if args.out else [])])


def _episode(args: argparse.Namespace) -> int:
    import uuid

    from liminal.orchestration import TaskSpecError, service
    from liminal.orchestration.procedure import ProcedureError, describe

    mode = "authored" if args.task or args.authored else "planner"
    run_id = args.resume or f"run-{uuid.uuid4().hex[:8]}"
    if mode == "planner" and not args.db and not args.plan_only:
        print("--db is required for planner episodes", file=sys.stderr)
        return 2
    if mode == "authored" and (args.resume or args.interrupt_after is not None):
        print("--task/--authored (V1) has no ledger: --resume and --interrupt-after need the planner path",
              file=sys.stderr)
        return 2
    # A grounded goal is frozen next to the ledger so a resume re-runs the SAME accepted obligations.
    frozen = Path(f"{args.db}.{run_id}.procedure.json") if args.db else None
    procedure = None
    if frozen is not None and args.resume and frozen.exists():
        procedure = json.loads(frozen.read_text())
    elif args.goal or args.demo:
        try:
            if args.demo:
                proposal, rows, extraction = service.ground_demo(args.scenario, args.demo, args.model,
                                                                 strategy=args.demo_strategy)
                print(f"demo: {extraction.get('demo')} ({extraction.get('strategy')} keyframes, "
                      f"{len(extraction.get('frames', []))} frames; {extraction.get('model')})")
            else:
                proposal, rows = service.ground(args.scenario, args.goal, args.model)
        except (ProcedureError, RuntimeError, ImportError, OSError, ValueError) as e:
            print(f"could not ground {'demo' if args.demo else 'goal'}: {e}", file=sys.stderr)
            return 2
        _print_breakdown(proposal, rows)
        if args.plan_only:
            return 0
        if not proposal.steps:
            print("nothing to run: no supported atomic skill in this goal", file=sys.stderr)
            return 2
        procedure = proposal.to_dict()
        if frozen is not None:
            frozen.parent.mkdir(parents=True, exist_ok=True)
            frozen.write_text(json.dumps(procedure, indent=2))
    elif args.plan_only:
        print("--plan-only needs --goal or --demo", file=sys.stderr)
        return 2
    options = service.RunOptions(
        args.scenario, mode=mode, model=args.model, procedure=procedure, task=args.task, db=args.db,
        run_id=run_id, resume=bool(args.resume), interrupt_after=args.interrupt_after,
        fail_first=args.fail_first, lie_first=args.lie_first, max_attempts=args.max_attempts,
        max_model_calls=args.max_model_calls, robot=args.robot, log_dir=args.log_dir, scene_state=args.scene_state,
        memory=_memory_options(args))
    try:
        prepared = service.prepare(options)
    except (TaskSpecError, ProcedureError, ValueError) as e:
        print(f"cannot start episode: {e}", file=sys.stderr)
        return 2
    if procedure is not None and not (args.goal or args.demo):
        _print_breakdown(prepared.procedure, describe(prepared.procedure, service.skill_world(prepared.scenario)))
    try:
        report = asyncio.run(service.execute(prepared))
    except KeyboardInterrupt:
        prepared.ctx.safety.estop()
        hint = f"; resume with --resume {run_id}" if mode == "planner" else ""
        print(f"e-stop: interrupted{hint}", file=sys.stderr)
        return 130
    except (RuntimeError, ImportError) as e:
        print(f"model unavailable: {e}", file=sys.stderr)
        return 2
    except ValueError as e:  # e.g. an authored skill the scenario cannot apply
        print(f"task does not fit scenario {prepared.scenario.name}: {e}", file=sys.stderr)
        return 2
    print(report.summary())
    print("NOTE: replay/mock run: scene evidence is simulated, not hardware verification.")
    print(f"event log: {prepared.log_path}")
    if not report.completed and mode == "planner":
        print(f"resume: liminal episode {args.scenario} --db {args.db} --resume {run_id}")
    return 0 if report.completed else 1


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

    e = sub.add_parser("episode", help="evidence-verified, resumable episode (replay/mock by default)")
    e.add_argument("scenario", help="replay scenario, e.g. configs/replay/two_blocks.yaml")
    e.add_argument("--db", default=None, help="SQLite memory path, e.g. runs/x.sqlite3 (planner path)")
    e.add_argument("--task", default=None, help="authored task file (V1): run its tree instead of the planner")
    e.add_argument("--goal", default=None, help='free-text goal, grounded into atomic skills')
    e.add_argument("--authored", action="store_true", help="run a --goal as a compiled reactive tree (no planner)")
    e.add_argument("--plan-only", action="store_true",
                   help="with --goal/--demo: print the atomic-skill breakdown and exit")
    e.add_argument("--demo", default=None, help="demonstration folder (V4), e.g. configs/demos/two_blocks_demo")
    e.add_argument("--demo-strategy", default="sampled", choices=["sampled", "manual"], help="demo keyframes")
    e.add_argument("--keyframes", default="event", choices=["event", "nominated", "none"], help="visual memory")
    e.add_argument("--max-keyframes", type=int, default=None)
    e.add_argument("--experience", default="off", choices=["off", "read", "readwrite"], help="cross-run memory")
    e.add_argument("--experience-db", default=None, help="default runs/experience.sqlite3")
    e.add_argument("--experience-frozen", action="store_true", help="read-only experience corpus")
    e.add_argument("--model", default="replay", choices=["replay", "claude"])
    e.add_argument("--interrupt-after", type=int, default=None, help="stop after N verified attempts")
    e.add_argument("--resume", default=None, metavar="RUN_ID", help="continue an existing run id from --db")
    e.add_argument("--log-dir", default="runs")
    e.add_argument("--robot", default="configs/robot/mock_dual.yaml")
    e.add_argument("--scene-state", default=None, help="replay scene state file (default: next to --db)")
    e.add_argument("--max-attempts", type=int, default=6)
    e.add_argument("--max-model-calls", type=int, default=None, help="default 20 (replay) / .env cap (claude)")
    e.add_argument("--fail-first", action="store_true", help="replay backend fails its first call")
    e.add_argument("--lie-first", action="store_true", help="replay backend claims success without acting")
    e.set_defaults(fn=_episode)

    ev = sub.add_parser("eval", help="compare harness configurations on replay (V6)")
    ev.add_argument("experiment", help="experiment YAML, e.g. configs/eval/orchestration.yaml")
    ev.add_argument("--out", default=None, help="output directory (default runs/eval/<name>)")
    ev.set_defaults(fn=_eval)

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
