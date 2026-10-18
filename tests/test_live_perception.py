import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from liminal.hardware.cameras import MockCamera
from liminal.perception import (
    FixtureVisionModel,
    InspectionProfile,
    InspectionRequest,
    PerceptionRuntime,
    SceneInspector,
    WorldState,
    inspect_live,
)
from liminal.perception.live import SnapshotCamera


def test_live_inspection_records_frames_consensus_and_blackboard_versions(tmp_path):
    response = {
        "predicates": {
            "lid_open": {"value": "false", "evidence": "the lid is visibly closed"},
        }
    }
    world = WorldState()
    inspector = SceneInspector(
        FixtureVisionModel([response, response]),
        {"top": SnapshotCamera(MockCamera("top"), tmp_path)},
        world,
        confirmations=2,
    )
    request = InspectionRequest(
        predicates=("lid_open",),
        cameras=("top",),
        definitions={"lid_open": "The lid is visibly raised."},
    )

    report = asyncio.run(inspect_live(inspector, request, samples=2, interval_s=0))

    assert report["arm_buses_connected"] is False
    assert report["arm_commands_issued"] is False
    assert report["passed"] is True
    assert report["samples"][0]["predicates"]["lid_open"]["value"] == "unknown"
    assert report["samples"][1]["predicates"]["lid_open"]["value"] == "false"
    captured = [Path(sample["images"]["top"]) for sample in report["samples"]]
    assert all(path.is_file() and tmp_path in path.parents for path in captured)
    assert report["final_blackboard"]["lid_open"]["value"] is False
    assert report["final_blackboard"]["lid_open"]["observation_version"] == 2


def test_live_inspection_aborts_after_model_timeout():
    class TimeoutModel:
        async def inspect(self, prompt, images):
            raise asyncio.TimeoutError

    inspector = SceneInspector(
        TimeoutModel(),
        {"top": MockCamera("top")},
        WorldState(),
        confirmations=1,
    )
    request = InspectionRequest(("lid_open",), timeout_s=0.1)

    report = asyncio.run(inspect_live(inspector, request, samples=3, interval_s=0))

    assert report["passed"] is False
    assert report["aborted_reason"] == "vision model timed out"
    assert len(report["samples"]) == 1


def test_live_cli_never_connects_or_commands_arms(monkeypatch, tmp_path):
    from liminal.ui.cli import _inspect_live

    response = {"predicates": {"lid_open": {"value": "true", "evidence": "open"}}}
    model = FixtureVisionModel([response])
    profile = InspectionProfile(
        cameras=("top",),
        definitions={"lid_open": "The lid is visibly raised."},
    )
    runtime = PerceptionRuntime(model, profile, confirmations=1)
    camera = MagicMock(name="top_camera")
    camera.name = "top"
    camera.read.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    arm = MagicMock(name="left_arm")
    world = WorldState()
    ctx = SimpleNamespace(
        arms={"left": arm},
        cameras={"top": camera},
        inspector=SceneInspector(model, {"top": camera}, world, confirmations=1),
    )
    monkeypatch.setattr("liminal.perception.config.load_perception_config", lambda path: runtime)
    monkeypatch.setattr("liminal.perception.vlm.warm_vision_model", lambda selected: None)
    monkeypatch.setattr("liminal.app.build_context", lambda *args, **kwargs: ctx)
    args = SimpleNamespace(
        perception="perception.yaml",
        predicate=None,
        robot="robot.yaml",
        samples=1,
        interval_s=0,
        output=str(tmp_path / "report.json"),
    )

    assert _inspect_live(args) == 0
    arm.connect.assert_not_called()
    arm.set_torque.assert_not_called()
    arm.send_action.assert_not_called()
