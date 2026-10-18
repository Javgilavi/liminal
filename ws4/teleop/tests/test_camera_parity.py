import numpy as np
import pytest
from camera_parity import StationCameras, apply_gains, rotate


def test_rotation_matches_cv2():
    cv2 = pytest.importorskip("cv2")
    img = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    assert np.array_equal(rotate(img, 180), cv2.rotate(img, cv2.ROTATE_180))
    assert np.array_equal(rotate(img, 90), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    assert np.array_equal(rotate(img, 270), cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))


def test_gains_clip_not_wrap():
    img = np.array([[[200, 250, 250]]], np.uint8)
    out = apply_gains(img, (1.0, 1.1, 1.229))
    assert out.tolist() == [[[200, 255, 255]]]


def test_station_specs_use_robot_sides():
    cams = StationCameras.load()
    assert cams.specs["left_wrist"].device == "/dev/haller_cam_wrist_right"  # robot left = haller right
    assert cams.specs["right_wrist"].device == "/dev/haller_cam_wrist_left"
    assert cams.specs["left_wrist"].rotation == 180 and cams.specs["head"].rotation == 0
    assert cams.specs["left_wrist"].color_gains == (1.0, 1.073, 1.23)
    img = np.full((480, 640, 3), 100, np.uint8)
    img[0, 0] = (10, 20, 30)
    out = cams.transform("left_wrist", img)
    assert out[-1, -1].tolist() == [10, 21, 36] and out.shape == (480, 640, 3)
    calls = []
    cams.apply_controls(setter=lambda dev, ctl: calls.append(dev) or ctl)
    assert sorted(calls) == ["/dev/haller_cam_wrist_left", "/dev/haller_cam_wrist_right"]
