"""Camera configuration and orientation checks without opening devices."""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from liminal.app import build_arms
from liminal.hardware.cameras import OpenCVCamera


def test_camera_config_preserves_device_and_capture_orientation():
    _, cameras = build_arms({"arms": {}, "cameras": {"wrist_a": {
        "type": "opencv", "index": "/dev/v4l/by-path/wrist-a", "width": 640, "height": 480,
        "fps": 15, "rotation": 180, "label": "Wrist A", "fourcc": "MJPG",
    }}})
    camera = cameras["wrist_a"]
    assert (camera.index, camera.width, camera.height, camera.fps) == ("/dev/v4l/by-path/wrist-a", 640, 480, 15)
    assert camera.rotation == 180
    assert camera.label == "Wrist A"
    assert camera.fourcc == "MJPG"


def test_rotation_applies_before_rgb_output_and_failed_reads_are_rejected(monkeypatch):
    cv = SimpleNamespace(ROTATE_90_CLOCKWISE=1, ROTATE_180=2, ROTATE_90_COUNTERCLOCKWISE=3,
                         COLOR_BGR2RGB=4, rotate=MagicMock(return_value="upright"),
                         cvtColor=MagicMock(return_value="rgb"))
    monkeypatch.setitem(sys.modules, "cv2", cv)
    camera = OpenCVCamera("wrist", "/fake", rotation=180)
    camera._cap = SimpleNamespace(read=MagicMock(return_value=(True, "raw")))
    assert camera.read() == "rgb"
    cv.rotate.assert_called_once_with("raw", cv.ROTATE_180)
    cv.cvtColor.assert_called_once_with("upright", cv.COLOR_BGR2RGB)
    camera._cap.read.return_value = (False, None)
    with pytest.raises(RuntimeError, match="returned no frame"):
        camera.read()
    with pytest.raises(ValueError, match="rotation"):
        OpenCVCamera("invalid", "/fake", rotation=45)


def test_linux_capture_selects_compressed_format_before_resolution(monkeypatch):
    capture = MagicMock()
    cv = SimpleNamespace(VideoCapture=MagicMock(return_value=capture), CAP_V4L2=1, CAP_ANY=0,
                         CAP_PROP_FOURCC=2, CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
                         CAP_PROP_FPS=5, CAP_PROP_BUFFERSIZE=6, VideoWriter_fourcc=lambda *chars: "MJPG")
    monkeypatch.setitem(sys.modules, "cv2", cv)
    camera = OpenCVCamera("wrist", "/dev/fake-camera", fourcc="MJPG")
    camera.connect()
    cv.VideoCapture.assert_called_once_with("/dev/fake-camera", cv.CAP_V4L2)
    assert capture.set.call_args_list[0].args == (cv.CAP_PROP_FOURCC, "MJPG")
    camera.disconnect()
    capture.release.assert_called_once()
