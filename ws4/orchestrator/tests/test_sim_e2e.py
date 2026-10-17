"""End to end on the laptop backend (a): stub planner, scripted sim skills, gate, ground-truth verifier, trial log."""

import json
import threading
import time

import pytest
from orchestrator.session import Session, SessionConfig, load_trial_log

GOAL = "put the apple and the yellow block in the box, then close the lid"


@pytest.fixture
def session(tmp_path, can_render):
    cfg = SessionConfig(realtime=False, seed=0, trial_log=str(tmp_path / "trials.jsonl"),
                        runs_dir=str(tmp_path / "runs"))
    s = Session(cfg)
    yield s
    s.runner.close()


def test_demo_goal_runs_to_completion_with_concurrency(session, tmp_path):
    proposal = session.plan(GOAL)
    assert proposal.result.ok, proposal.result.errors
    assert proposal.to_json()["concurrent_pairs"]
    result = session.execute(proposal.result.plan)
    assert result.status == "succeeded", result.failure
    assert (result.steps_done, result.steps_total, result.interventions) == (3, 3, 0)
    assert result.concurrency["both_arms_moving_ticks"] > 60  # both arms really moved at the same time
    obs = session.robot.get_observation()
    for check in ({"in_box": "apple"}, {"in_box": "yellow_block"}, {"lid_on_box": True}):
        assert session.sim.predicate(check, obs)[0], check
    # The attempt is in the trial log and in trial_log.py's summary table.
    tl = load_trial_log()
    trials = tl.load(tmp_path / "trials.jsonl")
    assert len(trials) == 1 and trials[0].success and trials[0].steps_done == 3
    table = tl.render(tl.summarize(trials), markdown=True)
    assert "pack_box" in table and "1/1 (100%)" in table
    run_dir = session.run_dir
    summary = json.loads((run_dir / "result.json").read_text())
    assert summary["result"]["status"] == "succeeded" and (run_dir / "plan_v1.json").exists()
    assert (run_dir / "events.jsonl").read_text().count("\n") > 5


def test_estop_holds_both_arms_and_resume_completes(session, tmp_path):
    """The run stays on this thread (it owns the GL context); a helper thread plays the operator."""
    proposal = session.plan(GOAL)
    seen = {}

    def operator():
        deadline = time.time() + 60
        while (session.executor is None or session.executor.t < 3.0) and time.time() < deadline:
            time.sleep(0.02)
        session.estop("test")
        time.sleep(0.3)
        ex = session.executor
        seen["state"] = ex.state
        steps, t = session.base_robot.steps, ex.t
        qpos = session.sim.env.data.qpos.copy()
        time.sleep(0.4)
        seen["frozen"] = (session.base_robot.steps == steps and ex.t == t
                          and bool((session.sim.env.data.qpos == qpos).all()))
        seen["targets"] = list(session.runner.active_targets())
        session.release()
        time.sleep(0.1)
        session.resume()

    helper = threading.Thread(target=operator, daemon=True)
    helper.start()
    result = session.execute(proposal.result.plan)
    helper.join(timeout=5)
    assert seen["state"] == "estopped"
    assert seen["frozen"]  # no physics step, no command, active time stopped
    assert seen["targets"] == []
    assert result.status == "succeeded" and result.interventions == 1
    trial = load_trial_log().load(tmp_path / "trials.jsonl")[-1]
    assert trial.steps_done == 3 and trial.interventions == 1 and not trial.success  # human help: not a clean success


def test_unreachable_sim_skill_fails_fast_and_replans(session):
    """open_lid is left-only; with the lid parked on the right side the left arm cannot reach it: the step fails
    without waiting for a timeout and the (stub) replan cannot fix it, so the run fails cleanly."""
    session.cfg.max_replans = 1
    session.cfg.auto_approve_replans = True
    proposal = session.plan("open the lid")
    assert proposal.result.ok
    result = session.execute(proposal.result.plan)
    assert result.status == "failed"
    msgs = " ".join(e["msg"] for e in session.events.recent())
    assert "unreachable" in msgs or "check failed" in msgs
