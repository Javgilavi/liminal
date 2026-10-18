import json
from types import SimpleNamespace

import calib
from conftest import CAL_FILE


class FakeBus:
    def __init__(self, port, cal, log):
        self.port, self.cal, self.log = port, cal, log

    def connect(self):
        self.log.append(("connect", self.port))

    def read_calibration(self):
        return {j: SimpleNamespace(**v) for j, v in self.cal.items()}

    def disconnect(self, disable_torque=True):
        self.log.append(("disconnect", disable_torque))

    def write_calibration(self, *a, **k):  # must never be called
        raise AssertionError("calib.py wrote to a servo")


def test_read_eeprom_is_read_only_and_keeps_torque(tmp_path):
    log = []
    cal = calib.read_eeprom("/dev/x", bus_factory=lambda p: FakeBus(p, CAL_FILE, log))
    assert cal == CAL_FILE
    assert ("disconnect", False) in log


def test_side_mapping_is_haller_operator_frame():
    assert calib.ARMS["left"][:3] == ("right", "/dev/haller_arm_right", "haller_right")
    assert calib.ARMS["right"][:3] == ("left", "/dev/haller_arm_left", "haller_left")
    assert calib.ARMS["left"][3] == "xlerobot_left"


def test_alias_copies_robot_sides(tmp_path):
    left = {j: dict(v, homing_offset=1) for j, v in CAL_FILE.items()}
    (tmp_path / "haller_left.json").write_text(json.dumps(left))
    (tmp_path / "haller_right.json").write_text(json.dumps(CAL_FILE))
    calib.alias(tmp_path)
    assert json.loads((tmp_path / "xlerobot_left.json").read_text()) == CAL_FILE  # robot left = haller right
    assert json.loads((tmp_path / "xlerobot_right.json").read_text()) == left
    assert "identical" in "\n".join(calib.status(tmp_path))


def test_write_new_backs_up_a_different_file(tmp_path):
    p = tmp_path / "haller_left.json"
    assert calib.write_new(p, CAL_FILE) == "written"
    assert calib.write_new(p, CAL_FILE) == "unchanged"
    other = {j: dict(v, range_min=v["range_min"] + 1) for j, v in CAL_FILE.items()}
    assert calib.write_new(p, other).startswith("replaced")
    assert len(list(tmp_path.glob("haller_left.json.bak-*"))) == 1
    assert calib.diff(json.loads(p.read_text()), other) == []


def test_snapshot(tmp_path, calib_dir):
    ds = tmp_path / "ds"
    out = calib.snapshot(ds, calib_dir)
    assert (out / "haller_left.json").exists() and (out / "README.txt").exists()
