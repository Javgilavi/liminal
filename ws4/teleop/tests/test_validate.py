import json

import validate
from conftest import CAL_FILE, make_haller_dataset


def _run(root, calib_dir, prompts=None):
    return validate.validate(root, prompts, calib_dir)


def test_clean_recording_passes(haller_ds, calib_dir):
    rep = _run(haller_ds, calib_dir, ["Open the box"])
    assert rep.errors == [], rep.errors
    assert rep.facts["rig"] == "real" and rep.facts["episodes"] == 2


def test_60fps_and_h264_and_size_fail(tmp_path, calib_dir):
    rep = _run(make_haller_dataset(tmp_path / "a", fps=60, codec="h264", hw=(720, 960)), calib_dir)
    msgs = " ".join(rep.errors)
    assert "fps 60" in msgs and "codec h264" in msgs and "720x960" in msgs


def test_action_equal_state_fails(tmp_path, calib_dir):
    rep = _run(make_haller_dataset(tmp_path / "a", action_eq_state=True), calib_dir)
    assert any("no teleop session" in e for e in rep.errors)


def test_dropped_video_frames_fail(tmp_path, calib_dir):
    rep = _run(make_haller_dataset(tmp_path / "a", short_video=3), calib_dir)
    assert any("left_wrist" in e and "encoder dropped" in e for e in rep.errors)


def test_task_outside_prompt_list_fails(haller_ds, calib_dir):
    rep = _run(haller_ds, calib_dir, ["Close the box"])
    assert any("not in the frozen prompt list" in e for e in rep.errors)


def test_calibration_mismatch_fails(tmp_path, calib_dir):
    other = {j: dict(v, homing_offset=v["homing_offset"] + 5) for j, v in CAL_FILE.items()}
    root = make_haller_dataset(tmp_path / "a", calib={"left": other, "right": CAL_FILE})
    rep = _run(root, calib_dir)
    assert any(e.startswith("left arm") and "homing_offset" in e for e in rep.errors)
    assert not any(e.startswith("right arm") for e in rep.errors)


def test_missing_calibration_file_fails(haller_ds, tmp_path):
    rep = _run(haller_ds, tmp_path)
    assert any("missing" in e and "calib.py harvest" in e for e in rep.errors)


def test_rate_drift(tmp_path, calib_dir):
    assert any("off 30" in e for e in _run(make_haller_dataset(tmp_path / "a", measured_hz=29.8), calib_dir).errors)
    assert any("off 30" in w for w in _run(make_haller_dataset(tmp_path / "b", measured_hz=29.88), calib_dir).warnings)


def test_sim_rig_is_flagged_not_failed(tmp_path):
    rep = _run(make_haller_dataset(tmp_path / "a", rig="sim"), None)
    assert rep.errors == [] and rep.facts["rig"] == "sim"
    assert any("pipeline only" in w for w in rep.warnings)


def test_prompts_yaml(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("prompts:\n  open: Open the box\n  pack: [Put the apple in the box, Close the box]\n")
    assert validate.load_prompts(p) == ["Open the box", "Put the apple in the box", "Close the box"]


def test_cli_exit_code(haller_ds, calib_dir, capsys):
    assert validate.main([str(haller_ds), "--calib-dir", str(calib_dir)]) == 0
    info = json.loads((haller_ds / "meta/info.json").read_text())
    info["fps"] = 20
    (haller_ds / "meta/info.json").write_text(json.dumps(info))
    assert validate.main([str(haller_ds), "--calib-dir", str(calib_dir)]) == 1
    assert "FAIL" in capsys.readouterr().out
