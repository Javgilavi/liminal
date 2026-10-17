"""FLUX backend execution loop for both embodiment options (PLAN.md), with a fake
model in place of the real checkpoint."""

import asyncio

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
    return FluxConfig(chunk_size=4, execute_steps=2, control_hz=1000.0, **kw)


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
