"""Executor logic on a fake robot and fake policies (no MuJoCo): DAG scheduling, concurrency, verification cadence,
timeouts, retries, replans, primitives, holds, e-stop and overrides."""

import dataclasses
import threading
import time
from collections import Counter

import numpy as np
from flux.backends import FakeBackend
from flux.contract import SO101_CONTRACT
from flux.dual_arm_runner import per_arm_target
from orchestrator.events import EventLog
from orchestrator.executor import Executor
from orchestrator.gate import GatedDualArmRunner
from orchestrator.plan import Plan, Step
from orchestrator.primitives import HOME_DEG, PrimitiveBackend
from orchestrator.skills import Skill, Vocabulary
from orchestrator.verifier import Verdict


def mini_vocab(timeout=2.0):
    def flux(name):
        return Skill(name, "flux", ("left", "right"), prompt=name.replace("_", " "), success_check=f"{name} ok",
                     timeout_s=timeout, verify_every_chunks=1, max_attempts=2)

    skills = {n: flux(n) for n in ("reach_a", "reach_b", "never")}
    skills["hold"] = Skill("hold", "primitive", ("left", "right"), timeout_s=60)
    skills["home"] = Skill("home", "primitive", ("left", "right"), timeout_s=10)
    return Vocabulary(skills)


class CountingVerifier:
    name = "counting"
    fast = True

    def __init__(self, done_after):
        self.done_after = done_after
        self.calls = Counter()

    def check(self, step, skill, obs):
        self.calls[step.id] += 1
        need = self.done_after.get(skill.name)
        done = need is not None and self.calls[step.id] >= need
        return Verdict(done, 1.0, f"check {self.calls[step.id]}", self.name)


def build(follow_robot, verifier, timeout=2.0, max_replans=1, on_replan=None):
    vocab = mini_vocab(timeout)
    robot = follow_robot(np.r_[HOME_DEG, HOME_DEG])
    fake, prim = FakeBackend(SO101_CONTRACT, velocity=0.0), PrimitiveBackend()
    backends, targets, by_target = {"fake": fake, "prim": prim}, [], {}
    for arm in ("left", "right"):
        for key in ("fake", "prim"):
            name = f"{arm}:{key}"
            targets.append(dataclasses.replace(per_arm_target(arm, key), name=name))
            by_target[name] = backends[key]
    runner = GatedDualArmRunner(robot, targets, backends, gate=None)
    ex = Executor(runner=runner, robot=robot, vocab=vocab, verifier=verifier,
                  resolve_target=lambda s, k: f"{s.arm}:{'prim' if k.kind == 'primitive' else 'fake'}",
                  backends=by_target, events=EventLog(), realtime=False, max_replans=max_replans,
                  on_replan=on_replan, max_run_s=60)
    return ex, robot


def plan(*steps, version=1):
    return Plan([Step(**s) for s in steps], goal="g", version=version)


def test_concurrent_dag_and_verify_cadence(follow_robot):
    verifier = CountingVerifier({"reach_a": 2, "reach_b": 3})
    ex, _ = build(follow_robot, verifier, timeout=10.0)
    p = plan(dict(id="a", arm="left", skill="reach_a", success_check="a", max_attempts=1),
             dict(id="b", arm="right", skill="reach_b", success_check="b", max_attempts=1),
             dict(id="c", arm="left", skill="reach_a", depends_on=["a", "b"], success_check="c", max_attempts=1))
    result = ex.run(p)
    assert result.status == "succeeded" and result.steps_done == 3
    a, b, c = (ex.runs[i] for i in "abc")
    assert a.t_start == b.t_start == 0.0  # both started on the first tick
    assert c.t_start >= max(a.t_end, b.t_end)
    assert verifier.calls == Counter({"a": 2, "b": 3, "c": 2})  # one check per chunk until it passes
    assert a.chunks == 2 and b.chunks == 3
    assert result.concurrency["both_arms_moving_ticks"] > 0


def test_timeout_retry_then_replan(follow_robot):
    verifier = CountingVerifier({"reach_a": 1})
    holder = {}

    def on_replan(state):
        assert state["failed"]["id"] == "x" and state["failed"]["attempts"] == 2
        holder["ex"].approve_replan(plan(dict(id="y", arm="left", skill="reach_a", success_check="y",
                                              max_attempts=1), version=2))

    ex, _ = build(follow_robot, verifier, timeout=1.0, on_replan=on_replan)
    holder["ex"] = ex
    result = ex.run(plan(dict(id="x", arm="left", skill="never", success_check="x", max_attempts=2)))
    assert result.status == "succeeded" and result.replans == 1
    assert "x" not in ex.runs and ex.runs["y"].status == "done"
    msgs = [e["msg"] for e in ex.events.recent()]
    assert any("attempt 1 failed (timeout" in m for m in msgs) and any("FAILED after 2" in m for m in msgs)


def test_no_replans_left_fails_the_run(follow_robot):
    ex, _ = build(follow_robot, CountingVerifier({}), timeout=0.5, max_replans=0)
    result = ex.run(plan(dict(id="x", arm="right", skill="never", success_check="x", max_attempts=1),
                         dict(id="z", arm="right", skill="reach_a", depends_on=["x"], success_check="z",
                              max_attempts=1)))
    assert result.status == "failed" and "timeout" in result.failure
    assert result.steps_done == 0 and result.steps_total == 2


def test_primitive_and_hold(follow_robot):
    verifier = CountingVerifier({"reach_b": 4})
    ex, robot = build(follow_robot, verifier, timeout=10.0)
    robot.state[:6] = 0.0  # left arm away from home
    p = plan(dict(id="home", arm="left", skill="home", max_attempts=1),
             dict(id="hold", arm="left", skill="hold", depends_on=["home"], hold_until=["b"], max_attempts=1),
             dict(id="b", arm="right", skill="reach_b", success_check="b", max_attempts=1))
    result = ex.run(p)
    assert result.status == "succeeded" and result.steps_total == 2  # holds are not task steps
    assert np.abs(robot.state[:6] - HOME_DEG).max() < 1e-6
    assert ex.runs["hold"].t_end >= ex.runs["b"].t_end


def test_estop_freezes_then_resume_and_override(follow_robot):
    ex, robot = build(follow_robot, CountingVerifier({}), timeout=1e6)
    p = plan(dict(id="x", arm="left", skill="never", success_check="x", max_attempts=1))
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault("r", ex.run(p)))
    thread.start()
    time.sleep(0.3)
    ex.estop("test")
    time.sleep(0.2)
    assert ex.state == "estopped" and ex.runs["x"].status == "interrupted"
    sent, t = robot.sent, ex.t
    time.sleep(0.3)
    assert robot.sent == sent and ex.t == t  # nothing sent, active time frozen
    ex.resume()  # ignored while latched
    time.sleep(0.1)
    assert ex.state == "estopped"
    ex.release()
    time.sleep(0.1)
    assert ex.state == "paused" and not ex.runner.guard.estopped
    ex.resume()
    time.sleep(0.2)
    assert ex.runs["x"].status == "running" and robot.sent > sent
    ex.override("x", True)
    thread.join(timeout=5)
    result = out["r"]
    assert result.status == "succeeded" and result.interventions == 2


def test_abort(follow_robot):
    ex, _ = build(follow_robot, CountingVerifier({}), timeout=1e6)
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault("r", ex.run(
        plan(dict(id="x", arm="left", skill="never", success_check="x", max_attempts=1)))))
    thread.start()
    time.sleep(0.2)
    ex.abort()
    thread.join(timeout=5)
    assert out["r"].status == "aborted" and ex.runner.active_targets() == []
