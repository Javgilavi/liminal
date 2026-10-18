"""FLUX backend execution loop for both embodiment options (PLAN.md), with a fake
model in place of the real checkpoint."""

import asyncio
import time

import pytest

from liminal.app import build_context
from liminal.core.types import Skill, Status
from liminal.perception import FixtureVisionModel, InspectionProfile
from liminal.policy.flux import FluxConfig, FluxPolicyBackend


class FakeFlux(FluxPolicyBackend):
    def load(self):
        self._model = object()

    def predict(self, history, instruction, target):
        self.predict_calls = getattr(self, "predict_calls", 0) + 1
        self.seen = (target, len(history[-1].state), sorted(history[-1].images))
        return [[float(i) for i in range(self.cfg.action_dim)]] * self.cfg.chunk_size


def _cfg(**kw):
    values = {"chunk_size": 4, "execute_steps": 2, "control_hz": 1000.0, **kw}
    return FluxConfig(**values)


def test_option_a_per_arm_drives_one_arm():
    backend = FakeFlux(_cfg(camera_map={"right": {"scene": "top", "wrist": "right_wrist"}}))
    ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=backend)
    r = asyncio.run(backend.execute(Skill("right", "pick_place", prompt="p"), ctx))
    assert r.status is Status.SUCCESS
    assert backend.seen == ("right", 6, ["scene", "wrist"])
    assert ctx.arms["right"].state == [0, 1, 2, 3, 4, 5]
    assert ctx.arms["left"].history == []


def test_option_b_bimanual_splits_12d_action():
    backend = FakeFlux(_cfg(action_dim=12, camera_map={
        "both": {"scene": "top", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}}))
    ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=backend)
    r = asyncio.run(backend.execute(Skill("both", "open_lid_and_insert", prompt="p"), ctx))
    assert r.status is Status.SUCCESS
    assert backend.seen == ("both", 12, ["left_wrist", "right_wrist", "scene"])
    assert ctx.arms["left"].state == [0, 1, 2, 3, 4, 5]
    assert ctx.arms["right"].state == [6, 7, 8, 9, 10, 11]


def test_action_dim_mismatch_fails_before_loading():
    backend = FakeFlux(_cfg(action_dim=12))
    ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=backend)
    r = asyncio.run(backend.execute(Skill("right", "pick_place", prompt="p"), ctx))
    assert r.status is Status.FAILURE and "12-D" in r.message
    assert backend._model is None


def test_option_a2_builds_one_instance_per_arm():
    ctx = build_context("configs/robot/mock_dual.yaml", backend="flux",
                        policy={"instances": "per_arm", "devices": {"left": "cuda:0", "right": "cuda:1"}})
    left, right = ctx.skills.arm_policies["left"], ctx.skills.arm_policies["right"]
    assert left is not right
    assert (left.cfg.device, right.cfg.device) == ("cuda:0", "cuda:1")
    assert ctx.skills.backend_for(Skill("left", "pick_place")) is left


def test_cancelled_prediction_quarantines_worker_and_discards_stale_actions():
    import threading

    class BlockingFlux(FakeFlux):
        def __init__(self):
            super().__init__(_cfg())
            self.entered = threading.Event()
            self.release = threading.Event()
            self.calls = 0

        def predict(self, history, instruction, target):
            self.calls += 1
            if self.calls == 1:
                self.entered.set()
                assert self.release.wait(5), "test did not release prediction"
                return [[9.0] * 6]
            return [[1.0] * 6]

    async def run():
        backend = BlockingFlux()
        ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=backend)
        first = asyncio.create_task(backend.execute(Skill("right", "pick_place", prompt="old"), ctx))
        try:
            assert await asyncio.to_thread(backend.entered.wait, 5)
            first.cancel()
            try:
                await first
            except asyncio.CancelledError:
                pass
            assert backend._unhealthy_reason is not None
            assert ctx.arms["right"].history == []
            backend.reset("right")
            second = asyncio.create_task(backend.execute(Skill("right", "pick_place", prompt="new"), ctx))
            # Quarantine prevents reuse even after the cancelled worker eventually exits.
            assert backend.calls == 1
            backend.release.set()
            assert (await asyncio.wait_for(second, 5)).status is Status.FAILURE
            assert backend.calls == 1
            assert ctx.arms["right"].history == []
        finally:
            backend.release.set()
    asyncio.run(run())


def test_vlm_terminates_flux_after_confirmed_chunk_and_checks_invariant(monkeypatch):
    async def inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline)
    before = {
        "predicates": {
            "object_placed": {"value": "false", "evidence": "object remains in the box"},
            "cable_clear": {"value": "true", "evidence": "cable is held away"},
        }
    }
    after = {
        "predicates": {
            "object_placed": {"value": "true", "evidence": "object is in the red cup"},
            "cable_clear": {"value": "true", "evidence": "cable is still held away"},
        }
    }
    model = FixtureVisionModel([before, before, after, after])
    backend = FakeFlux(_cfg())
    profile = InspectionProfile(
        definitions={
            "object_placed": "The removed object is visibly resting inside the red cup.",
            "cable_clear": "The cable is held clear of the working arm and box opening.",
        },
        timeout_s=0.2,
        retry_delay_s=0,
    )
    ctx = build_context(
        "configs/robot/mock_dual.yaml",
        policy_backend=backend,
        vision_model=model,
        inspection=profile,
    )
    ctx.world.observe(
        "object_placed",
        True,
        source="previous_skill_run",
        evidence="stale result from an earlier attempt",
    )
    skill = Skill("right", "remove_to_red_cup", prompt="remove the object", success="object_placed")

    with ctx.watch_invariant("cable_clear"):
        result = asyncio.run(backend.execute(skill, ctx))

    assert result.status is Status.SUCCESS
    assert backend.predict_calls == 1
    assert len(ctx.arms["right"].history) == backend.cfg.execute_steps
    assert ctx.arms["right"].stopped
    assert all("object_placed" in prompt and "cable_clear" in prompt for prompt, _ in model.calls)


def test_vlm_blocks_flux_when_invariant_is_confirmed_false():
    response = {
        "predicates": {
            "object_placed": {"value": "false", "evidence": "object remains in the box"},
            "cable_clear": {"value": "false", "evidence": "cable crosses the work area"},
        }
    }
    model = FixtureVisionModel([response, response])
    backend = FakeFlux(_cfg())
    ctx = build_context(
        "configs/robot/mock_dual.yaml",
        policy_backend=backend,
        vision_model=model,
        inspection=InspectionProfile(
            definitions={
                "object_placed": "The removed object is visibly resting inside the red cup.",
                "cable_clear": "The cable is held clear of the working arm and box opening.",
            },
            timeout_s=0.2,
            retry_delay_s=0,
        ),
    )
    skill = Skill("right", "remove_to_red_cup", prompt="remove the object", success="object_placed")

    with ctx.watch_invariant("cable_clear"):
        result = asyncio.run(backend.execute(skill, ctx))

    assert result.status is Status.FAILURE
    assert "cable_clear" in result.message
    assert not hasattr(backend, "predict_calls")
    assert ctx.arms["right"].history == [] and ctx.arms["right"].stopped


def test_vlm_invariant_failure_after_motion_prevents_next_chunk(monkeypatch):
    async def inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline)
    safe = {
        "predicates": {
            "object_placed": {"value": "false", "evidence": "object remains in the box"},
            "cable_clear": {"value": "true", "evidence": "cable is held away"},
        }
    }
    dropped = {
        "predicates": {
            "object_placed": {"value": "false", "evidence": "object remains in the box"},
            "cable_clear": {"value": "false", "evidence": "cable fell into the work area"},
        }
    }
    backend = FakeFlux(_cfg())
    ctx = build_context(
        "configs/robot/mock_dual.yaml",
        policy_backend=backend,
        vision_model=FixtureVisionModel([safe, safe, dropped, dropped]),
        inspection=InspectionProfile(
            definitions={
                "object_placed": "The removed object is visibly resting inside the red cup.",
                "cable_clear": "The cable is held clear of the working arm and box opening.",
            },
            timeout_s=0.2,
            retry_delay_s=0,
        ),
    )
    skill = Skill("right", "remove_to_red_cup", prompt="remove the object", success="object_placed")

    with ctx.watch_invariant("cable_clear"):
        result = asyncio.run(backend.execute(skill, ctx))

    assert result.status is Status.FAILURE
    assert "cable_clear" in result.message
    assert backend.predict_calls == 1
    assert len(ctx.arms["right"].history) == backend.cfg.execute_steps
    assert ctx.arms["right"].stopped


class SlowFlux(FakeFlux):
    def predict(self, history, instruction, target):
        self.predict_calls = getattr(self, "predict_calls", 0) + 1
        time.sleep(0.05)
        return [[0.0] * self.cfg.action_dim] * self.cfg.chunk_size


def test_inference_timeout_freezes_and_quarantines_backend():
    backend = SlowFlux(_cfg(inference_timeout_s=0.005))
    ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=backend)
    skill = Skill("right", "pick_place", prompt="pick and place")

    async def run_twice():
        return await backend.execute(skill, ctx), await backend.execute(skill, ctx)

    timed_out, retried = asyncio.run(run_twice())

    assert timed_out.status is Status.TIMEOUT
    assert retried.status is Status.FAILURE and "quarantined" in retried.message
    assert backend.predict_calls == 1
    assert ctx.arms["right"].history == []
    assert ctx.arms["right"].stopped


@pytest.mark.parametrize(
    "chunk",
    [
        [[0.0] * 6] * 3,
        [[0.0] * 5] * 4,
        [[0.0] * 6, [0.0] * 6, [0.0] * 6, [0.0, 0.0, 0.0, 0.0, 0.0, float("nan")]],
    ],
)
def test_malformed_action_chunk_fails_before_motion(chunk):
    class MalformedFlux(FakeFlux):
        def predict(self, history, instruction, target):
            return chunk

    backend = MalformedFlux(_cfg())
    ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=backend)

    result = asyncio.run(backend.execute(Skill("right", "pick_place", prompt="move"), ctx))

    assert result.status is Status.FAILURE
    assert "invalid FLUX output" in result.message
    assert ctx.arms["right"].history == []
    assert ctx.arms["right"].stopped


def test_cancellation_freezes_arm_and_stops_action_stream():
    backend = FakeFlux(_cfg(control_hz=100.0))
    ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=backend)
    skill = Skill("right", "pick_place", prompt="keep moving", success="not_yet_complete")

    async def cancel_during_actions():
        task = asyncio.create_task(backend.execute(skill, ctx))
        while not ctx.arms["right"].history:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        actions_after_cancel = len(ctx.arms["right"].history)
        await asyncio.sleep(0.03)
        return actions_after_cancel

    actions_after_cancel = asyncio.run(cancel_during_actions())

    assert actions_after_cancel > 0
    assert len(ctx.arms["right"].history) == actions_after_cancel
    assert ctx.arms["right"].stopped


@pytest.mark.parametrize(
    "kwargs",
    [
        {"execute_steps": 0},
        {"chunk_size": 2, "execute_steps": 3},
        {"control_hz": float("inf")},
        {"inference_timeout_s": 0},
    ],
)
def test_flux_config_rejects_unsafe_execution_values(kwargs):
    with pytest.raises(ValueError):
        FluxConfig(**kwargs)


@pytest.mark.parametrize("profile", ["flux3_so101.yaml", "showcase_registry.yaml"])
def test_dual_arm_profiles_route_independent_instances(profile):
    ctx = build_context("configs/robot/mock_dual.yaml", backend="flux",
                        policy=f"configs/policy/{profile}")
    left = ctx.skills.backend_for(Skill("left", "pick_place"))
    right = ctx.skills.backend_for(Skill("right", "pick_place"))
    assert left is not right
    assert left._execution_lock is not right._execution_lock
    assert left._infer_lock is not right._infer_lock
    assert left.cfg.action_dim == right.cfg.action_dim == 6


def test_per_arm_predictions_can_overlap(monkeypatch):
    import threading

    barrier = threading.Barrier(2, timeout=5)

    class ConcurrentFlux(FakeFlux):
        def predict(self, history, instruction, target):
            # A shared inference/execution lock would prevent the second arm arriving.
            barrier.wait()
            return super().predict(history, instruction, target)

    monkeypatch.setattr("liminal.app.build_policy", lambda *args: ConcurrentFlux(_cfg()))
    ctx = build_context("configs/robot/mock_dual.yaml", backend="flux",
                        policy={"instances": "per_arm"})

    async def run():
        return await asyncio.gather(*(
            ctx.skills.backend_for(Skill(arm, "pick_place")).execute(
                Skill(arm, "pick_place", prompt=arm), ctx)
            for arm in ("left", "right")
        ))

    results = asyncio.run(run())
    assert all(result.status is Status.SUCCESS for result in results)
    for arm in ("left", "right"):
        backend = ctx.skills.arm_policies[arm]
        assert backend.seen[0] == arm
        assert ctx.arms[arm].history
