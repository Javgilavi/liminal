from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import backend
import pytest
import sim_take
import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@dataclass
class FakeRecorder:
    telemetry: object
    vcodec: str = "h264"


def test_codec_patch_is_idempotent():
    srv = SimpleNamespace(DatasetRecorder=FakeRecorder)
    backend.patch_recorder_codec(srv, "libsvtav1")
    backend.patch_recorder_codec(srv, "libsvtav1")
    assert srv.DatasetRecorder(telemetry=None).vcodec == "libsvtav1"
    backend.patch_recorder_codec(srv, "h264")
    assert srv.DatasetRecorder(telemetry=None).vcodec == "h264"


def test_session_rate_patch():
    srv = SimpleNamespace(human_teleop=SimpleNamespace(_hz_override=None))
    backend.patch_session_rate(srv, backend.SESSION_HZ)
    assert srv.human_teleop._hz_override == pytest.approx(30.08)
    assert round(backend.SESSION_HZ) == 30


@pytest.mark.parametrize("name", ["station-sim.yaml", "station-real.yaml"])
def test_configs_record_the_flux_cameras(name):
    cfg = yaml.safe_load((CONFIGS / name).read_text())
    rec = {c["dataset_key"]: c for c in cfg["cameras"] if c.get("record", True)}
    assert set(rec) == {"head", "left_wrist", "right_wrist"}
    for c in rec.values():
        assert (c["width"], c["height"], c["fps"]) == (640, 480, 30)
    assert [a["id"] for a in cfg["arms"]] == ["left", "right"]


def test_real_config_ports_and_calibration_follow_haller_ids():
    cfg = yaml.safe_load((CONFIGS / "station-real.yaml").read_text())
    arms = {a["id"]: a for a in cfg["arms"]}
    assert arms["left"]["port"] == "/dev/haller_arm_left" and arms["left"]["calibration_id"] == "haller_left"
    assert arms["right"]["port"] == "/dev/haller_arm_right" and arms["right"]["calibration_id"] == "haller_right"
    wrists = {c["arm_id"]: c["index_or_path"] for c in cfg["cameras"] if c["role"] == "wrist"}
    assert wrists == {"left": "/dev/haller_cam_wrist_left", "right": "/dev/haller_cam_wrist_right"}
    assert cfg["collision"]["enabled"] is False
    # haller left (robot right) sits at station y = -0.1366 -> haller x = -0.1366
    assert cfg["collision"]["mounts"]["left"]["pos"][0] == pytest.approx(-0.1366)


def test_configs_load_in_haller_if_available():
    hb = Path.home() / "haller_ws/hmi/backend"
    if not (hb / "haller_hmi/config.py").exists():
        pytest.skip("haller_ws checkout not present")
    import sys

    sys.path.insert(0, str(hb))
    try:
        from haller_hmi.config import load_config
    except Exception as e:  # haller deps (yaml only here) missing in this env
        pytest.skip(f"haller_hmi.config not importable: {e}")
    for name in ("station-sim.yaml", "station-real.yaml"):
        cfg = load_config(CONFIGS / name)
        keys = sorted(c.dataset_feature_key for c in cfg.cameras if c.record)
        assert keys == ["head", "left_wrist", "right_wrist"]


def test_sim_take_wave_starts_and_ends_at_rest():
    n = 90
    assert sim_take.wave(0, n, 0.1) == pytest.approx(0.0)
    assert sim_take.wave(n - 1, n, 0.1) == pytest.approx(0.0, abs=1e-12)
    assert max(sim_take.wave(i, n, 0.1) for i in range(n)) == pytest.approx(0.1, rel=1e-2)
