"""Command line: ``cd ws4 && python -m orchestrator <command>``.

  hmi        open the localhost HMI (plan, approve, run, e-stop)
  run        plan a goal, print it, and run it headless after confirmation (--yes to skip)
  plan       plan a goal and print the plan JSON (no robot motion)
  validate   validate a plan JSON file against the vocabulary
  schema     print the plan JSON schema (for other planners, e.g. WS3)
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from pathlib import Path

from . import WS4  # noqa: F401  (sets sys.path for flux, sim, kinematics)


def _session_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("backend")
    g.add_argument("--env", choices=("sim", "xlerobot"), default="sim")
    g.add_argument("--policy", choices=("scripted", "fake", "remote"), default=None,
                   help="scripted = sim stand-in (default in sim), remote = FLUX policy server (default on the robot)")
    g.add_argument("--planner", choices=("stub", "claude"), default="stub")
    g.add_argument("--verifier", choices=("auto", "truth", "claude", "shadow", "manual"), default="auto")
    g.add_argument("--planner-model", default="claude-opus-5-5")
    g.add_argument("--planner-effort", default="medium", choices=("low", "medium", "high", "xhigh", "max"))
    g.add_argument("--verifier-wrist", action="store_true", help="also send the arm's wrist frame to the verifier")
    g.add_argument("--budget-usd", type=float, default=5.0, help="Claude spend limit for this session")
    g.add_argument("--server", default="http://127.0.0.1:8765", help="FLUX policy server (ssh -L tunnel to the pod)")
    g.add_argument("--token-file", default=None)
    g.add_argument("--package", default="so101", help="policy server package for both arms")
    g.add_argument("--left-package", default=None)
    g.add_argument("--right-package", default=None)
    g.add_argument("--sync-inference", action="store_true", help="block the control loop during inference")
    s = p.add_argument_group("sim")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--lid-on-box", action="store_true")
    s.add_argument("--no-randomize", action="store_true")
    s.add_argument("--image-size", default="240x320", help="HxW of the sim cameras (480x640 for Claude checks)")
    r = p.add_argument_group("real robot (only when it is connected)")
    for flag in ("--robot-id", "--left-port", "--right-port", "--head-cam", "--left-wrist-cam", "--right-wrist-cam"):
        r.add_argument(flag, default="xlerobot" if flag == "--robot-id" else "")
    r.add_argument("--calibration-limits", action="store_true")
    e = p.add_argument_group("execution and logging")
    e.add_argument("--no-gate", action="store_true", help="disable the inter-arm collision gate")
    e.add_argument("--fast", action="store_true", help="sim: run as fast as possible instead of 30 Hz")
    e.add_argument("--max-replans", type=int, default=1)
    e.add_argument("--auto-approve-replans", action="store_true")
    e.add_argument("--trial-log", default=None,
                   help="default: runs/trials_sim.jsonl in sim, ws4/eval/trials.jsonl on the robot")
    e.add_argument("--task", default="pack_box")
    e.add_argument("--condition", default="zero_shot", choices=("zero_shot", "post_training"))
    e.add_argument("--variant", default=None)
    e.add_argument("--checkpoint", default=None)
    e.add_argument("--layout", default=None)
    e.add_argument("--operator", default="")
    e.add_argument("--vocabulary", default=None, help="skills YAML (default ws4/orchestrator/skills.yaml)")


def _config(a: argparse.Namespace):
    from .session import SessionConfig

    h, w = (int(x) for x in a.image_size.lower().split("x"))
    policy = a.policy or ("scripted" if a.env == "sim" else "remote")
    return SessionConfig(
        env=a.env, policy=policy, planner=a.planner, verifier=a.verifier, planner_model=a.planner_model,
        planner_effort=a.planner_effort, verifier_wrist=a.verifier_wrist, budget_usd=a.budget_usd, server=a.server,
        token_file=a.token_file, package=a.package, left_package=a.left_package, right_package=a.right_package,
        async_inference=False if a.sync_inference else None, seed=a.seed, image_size=(h, w),
        randomize=not a.no_randomize, lid_on_box=a.lid_on_box, robot_id=a.robot_id, left_port=a.left_port,
        right_port=a.right_port, head_cam=a.head_cam, left_wrist_cam=a.left_wrist_cam,
        right_wrist_cam=a.right_wrist_cam, calibration_limits=a.calibration_limits, gate=not a.no_gate,
        realtime=not a.fast, max_replans=a.max_replans, auto_approve_replans=a.auto_approve_replans,
        trial_log=a.trial_log, task=a.task, condition=a.condition, variant=a.variant, checkpoint=a.checkpoint,
        layout=a.layout, operator=a.operator, vocabulary=a.vocabulary,
    )


def cmd_hmi(a: argparse.Namespace) -> int:
    from .hmi.server import make_server
    from .session import Session

    session = Session(_config(a), threaded=True)
    server = make_server(session, "127.0.0.1", a.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"HMI on {url}  (Ctrl+C to quit; the e-stop is the red button or the spacebar)", flush=True)
    if a.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.estop("HMI shutting down")
        session.close()
        server.server_close()
    return 0


def _print_plan(proposal) -> None:
    data = proposal.to_json()
    print(json.dumps(data.get("plan", {}), indent=2))
    print(f"\nsource {data['source']}, {data['attempts']} attempt(s), {data['latency_s']} s, "
          f"USD {data['cost_usd']:.4f}")
    if data["errors"]:
        print("ERRORS:\n- " + "\n- ".join(data["errors"]))
    else:
        print(f"concurrent pairs: {data.get('concurrent_pairs') or 'none'}")


def cmd_plan(a: argparse.Namespace) -> int:
    from .session import Session

    session = Session(_config(a))
    proposal = session.plan(a.goal)
    _print_plan(proposal)
    if a.out and proposal.result.ok:
        Path(a.out).write_text(proposal.result.plan.dumps())
        print(f"wrote {a.out}")
    return 0 if proposal.result.ok else 1


def cmd_run(a: argparse.Namespace) -> int:
    from .session import Session

    cfg = _config(a)
    cfg.echo_events = True
    session = Session(cfg)
    if a.plan_file:
        proposal = session.load_plan_json(Path(a.plan_file).read_text(), a.goal or "")
    else:
        proposal = session.plan(a.goal)
    _print_plan(proposal)
    if not proposal.result.ok:
        return 1
    if not a.yes and input("\nApprove and run this plan? [y/N] ").strip().lower() != "y":
        print("not approved")
        return 1
    try:
        result = session.execute(proposal.result.plan)
    except KeyboardInterrupt:
        session.estop("Ctrl+C")
        return 130
    finally:
        session.runner.close()
    print(json.dumps(session.last_result["result"], indent=2))
    print(f"run directory: {session.run_dir}")
    return 0 if result.status == "succeeded" else 2


def cmd_validate(a: argparse.Namespace) -> int:
    from .plan import fill_defaults, parse_plan, validate_plan
    from .skills import load_vocabulary

    vocab = load_vocabulary(a.vocabulary)
    plan, errors = parse_plan(Path(a.file).read_text())
    if plan is not None:
        fill_defaults(plan, vocab)
        errors = validate_plan(plan, vocab)
    print("valid" if not errors else "INVALID:\n- " + "\n- ".join(errors))
    return 0 if not errors else 1


def cmd_schema(a: argparse.Namespace) -> int:
    from .plan import plan_json_schema
    from .skills import load_vocabulary

    print(json.dumps(plan_json_schema(load_vocabulary(a.vocabulary)), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m orchestrator", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("hmi", help="open the localhost HMI")
    _session_args(p)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--open", action="store_true", help="open the page in the default browser")
    p.set_defaults(func=cmd_hmi)
    p = sub.add_parser("run", help="plan and run a goal headless")
    _session_args(p)
    p.add_argument("--goal", default="put the apple and the yellow block in the box, then close the lid")
    p.add_argument("--plan-file", default=None, help="run this plan JSON instead of planning")
    p.add_argument("--yes", action="store_true", help="approve without asking")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("plan", help="plan a goal and print it")
    _session_args(p)
    p.add_argument("--goal", default="put the apple and the yellow block in the box, then close the lid")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("validate", help="validate a plan JSON file")
    p.add_argument("file")
    p.add_argument("--vocabulary", default=None)
    p.set_defaults(func=cmd_validate)
    p = sub.add_parser("schema", help="print the plan JSON schema")
    p.add_argument("--vocabulary", default=None)
    p.set_defaults(func=cmd_schema)
    a = parser.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
