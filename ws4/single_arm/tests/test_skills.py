"""Per-skill eval tooling: the skill/arm table, the robot-right arm's station keep-out, the mirrored home check, the
build split rules and eval_skill.sh's commands for both arms."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent.parent
LAYOUTS = HERE / "layouts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import skills  # noqa: E402
from keepout import ZONES, KeepOut  # noqa: E402


def test_pen_skill_env_is_the_pen_kit():
    lines = skills.shell_env("pen").splitlines()
    env = dict(line.split("=", 1) for line in lines if not line.startswith("SKILL_STEPS"))
    assert env["SKILL_PORT"] == "/dev/haller_arm_right" and env["SKILL_ROBOT_ID"] == "haller_right"
    assert env["SKILL_WORKSPACE"] == "ws4/single_arm/calibration/haller_right_workspace_final.json"
    assert env["SKILL_HOME_REF"] == "ws4/single_arm/frames/reference_2026-10-17_night"
    assert env["SKILL_LAYOUTS"] == "'A B C D A'" and env["SKILL_ARM"] == "left" and env["SKILL_VALID"] == "1"


def test_arms_follow_the_mirrored_haller_naming():
    left, right = skills.ARMS["left"], skills.ARMS["right"]
    assert (left.port, left.robot_id, left.columns, left.wrist_key) == \
        ("/dev/haller_arm_right", "haller_right", "right", "observation.images.right_wrist")  # fmt: skip
    assert (right.port, right.robot_id, right.columns, right.wrist_key) == \
        ("/dev/haller_arm_left", "haller_left", "left", "observation.images.left_wrist")  # fmt: skip
    assert left.wrist_cam.endswith("usb-0:7.2:1.0-video-index0")
    assert right.wrist_cam.endswith("usb-0:7.3:1.0-video-index0")
    assert (left.other_port, right.other_port) == (right.port, left.port)
    assert skills.SKILLS["box"].arm == "right"
    assert {skills.SKILLS[s].arm for s in ("green", "cutter", "pen")} == {"left"}


def test_right_arm_station_keepout_mirrors_the_left_one():
    left, right = KeepOut.for_haller("left"), KeepOut.for_haller("right")
    assert np.isclose(left.base_pos[1], -right.base_pos[1]) and right.other_base_pos == left.base_pos
    assert (left.side, right.side) == (1, -1)
    right.other_q = (0.1, -102.7, 96.0, -49.2, -11.6, 2.6)  # the robot-left arm folded at rest
    home = np.array([0, 0, 0, 0, 0, 100.0])
    assert right.depths(home).sum() == 0
    assert right.depths(np.array([3.0, -102.6, 96.7, 78.3, 12.9, 1.4])).sum() == 0  # its folded park pose
    reach = np.array([-60.0, 0, 0, 80.0, 0, 50])  # pan toward the parked left arm, wrist down
    assert right.refuse(reach, home)[0] and not right.refuse(home, reach)[0]
    assert right.depths(reach)[ZONES.index("other_arm")] > 0


def test_home_check_looks_for_the_right_arm_on_the_image_right():
    import home_check as hc
    from placement import read_rgb

    ref_dir = HERE / "frames" / "reference_2026-10-17_night"
    if not ref_dir.exists():
        pytest.skip("ws4-vr2's reference frames are not in this checkout")
    rh, rw, state, _ = hc.load_refs(ref_dir)
    head = read_rgb(LAYOUTS / "A_ep020_head.jpg")
    flip = lambda img: np.ascontiguousarray(img[:, ::-1])  # noqa: E731
    assert np.array_equal(hc.green_mask(flip(head), "right"), flip(hc.green_mask(head, "left")))
    ok_left, _, _ = hc.home_check(head, rw, state, rh, rw, state, side="left")
    ok_right, lines, _ = hc.home_check(flip(head), rw, state, flip(rh), rw, state, side="right")
    assert ok_left and ok_right, lines


def test_layout_refs_index_gives_the_home_check_its_joints(tmp_path):
    import shutil

    import home_check as hc

    for name in ("A_ep020_head.jpg", "A_ep020_wrist.jpg", "index.json"):
        shutil.copy(LAYOUTS / name, tmp_path / name)
    _, _, state, name = hc.load_refs(tmp_path)
    assert name == "A_ep020_head.jpg" and np.allclose(state[:5], [0, 2.5, 1.8, 0.2, 0], atol=0.1)
    (tmp_path / "A_ep020_wrist.jpg").unlink()
    with pytest.raises(FileNotFoundError):  # the runner catches this; SystemExit would kill it with the arm at home
        hc.load_refs(tmp_path)


def test_build_split_without_ws4_split_holds_out_the_last_three_good_takes(tmp_path):
    import build_skill

    review = {"episodes": {"4": {"status": "reject"}, "7": {"status": "keep"}, "9": {"status": "reject"}}}
    (tmp_path / "review.json").write_text(json.dumps(review))
    split = build_skill.split_of(tmp_path, list(range(10)))
    assert split["provisional"] and split["drop"] == [4, 9]
    assert split["heldout"] == {"H": [6, 7, 8]} and split["train"] == [0, 1, 2, 3, 5]


def _skill_dir(tmp_path, ok=True):
    box = tmp_path / "box"
    box.mkdir()
    (box / "skill.json").write_text(json.dumps({
        "skill": "box", "arm": "right", "task": "open the box by pulling its handle", "workspace": str(box / "ws.json"),
        "layouts": str(box / "layouts"), "eval_layouts": "H H H", "validation": {"ok": ok}}))  # fmt: skip
    return tmp_path


def _dry_run(skill_dir, *env_items):
    env = {**os.environ, "WS4_SKILL_DIR": str(skill_dir), "SKILL": "box", "DRY_RUN": "1", "POD_HOST": "h",
           "POD_PORT": "1", "LORA": "lora_box_5000_raw", "LORA_CHECKPOINT": "test", **dict(env_items)}  # fmt: skip
    return subprocess.run(["bash", str(HERE / "eval_skill.sh"), "session"], cwd=REPO, env=env, capture_output=True,
                          text=True)  # fmt: skip


def test_eval_skill_runs_the_box_on_the_right_arm(tmp_path):
    out = _dry_run(_skill_dir(tmp_path))
    assert out.returncode == 0, out.stderr
    cmd = [line for line in out.stdout.splitlines() if line.startswith("python ")][-1]
    for part in ("--port /dev/haller_arm_left --robot-id haller_left",
                 "--other-arm-port /dev/haller_arm_right --other-arm-id haller_right", "usb-0:7.3:1.0-video-index0",
                 "--task-side right", "--trial-task open_box",
                 "--other-arm-lo 3 -83.6 81.9 82.8 -10.2 0.9 --other-arm-hi 3 -83.6 81.9 82.8 -10.2 0.9",
                 "--steps reach the handle grasp the handle open the box",
                 "--task open the box by pulling its handle", "--variant LoRA-5k-raw", "--layouts H H H"):  # fmt: skip
        assert part in cmd, part


def test_eval_skill_refuses_a_skill_whose_demos_the_keepout_refused(tmp_path):
    out = _dry_run(_skill_dir(tmp_path, ok=False))
    assert out.returncode == 1 and "refused demonstrated motion" in out.stderr
    assert _dry_run(tmp_path, ("ALLOW_KEEPOUT_REFUSALS", "1")).returncode == 0


def test_green_without_a_built_parked_range_does_not_run(tmp_path):
    # The box arm holds the lid during the pen-arm skills: its range can only come from the green demos.
    green = tmp_path / "green"
    green.mkdir()
    (green / "skill.json").write_text(json.dumps({
        "skill": "green", "arm": "left", "task": "t", "workspace": "w", "layouts": "l", "validation": {"ok": True}}))
    out = _dry_run(tmp_path, ("SKILL", "green"))
    assert out.returncode != 0 and "no parked-arm range" in out.stderr
    rec = json.loads((green / "skill.json").read_text())
    rec["parked_other_arm"] = {"min": [10, -40, 20, 30, -5, 0], "max": [16, -34, 27, 41, 3, 5]}
    (green / "skill.json").write_text(json.dumps(rec))
    out = _dry_run(tmp_path, ("SKILL", "green"))
    assert out.returncode == 0, out.stderr
    assert "--other-arm-lo 10 -40 20 30 -5 0 --other-arm-hi 16 -34 27 41 3 5" in out.stdout


def test_a_skill_built_without_references_does_not_run(tmp_path):
    skill_dir = _skill_dir(tmp_path)
    rec = json.loads((skill_dir / "box" / "skill.json").read_text())
    (skill_dir / "box" / "skill.json").write_text(json.dumps({**rec, "refs": False}))
    out = _dry_run(skill_dir)
    assert out.returncode == 1 and "--no-refs" in out.stderr


def test_envelope_gap_samples_inside_each_voxel():
    import build_skill

    a, b = np.zeros(3), np.array([1.0, 0, 0])
    d = build_skill.point_segment(np.array([[0.5, 1, 0], [-1, 0, 0], [2, 0, 1]]), a, b)
    assert np.allclose(d, [1, 1, np.sqrt(2)])
    ws = KeepOut.for_haller("left")
    rng = np.random.default_rng(0)
    ws.build_envelope(rng.uniform([-30, -20, -40, 0, -20, 0], [60, 70, 50, 80, 50, 100], size=(200, 6)))
    box_arm = np.array([12.1, -13.8, 12.3, 37.5, 29.9, 0.9])  # holding the lid during the green takes
    gap = build_skill.envelope_gap(ws, [box_arm])
    assert gap.shape == (len(ws.env_cells),)
    centre_only = build_skill.envelope_gap(ws, [box_arm], samples=1)
    assert np.all(gap <= centre_only + 1e-12)  # points inside a voxel can only come closer than its centre


GREEN = HERE / "skills" / "green"
GREEN_HOLD = [12.13, -13.80, 12.26, 37.54, 29.93, 0.87]  # box arm during the green and cutter takes
BOX_TAKE_74_END = [4.2, -31.4, 48.4, -2.3, 67.6, 0.9]  # a box-skill end pose that intrudes into green's envelope
RED_PEN_HOLD = [0.88, 15.12, -5.85, 32.62, 14.64, 0.87]


def _green_gate():
    if not (GREEN / "demo_poses.npz").exists():
        pytest.skip("skills/green has no gate yet (gate.py write green)")
    import gate

    info = json.loads((GREEN / "skill.json").read_text())
    return gate, KeepOut.load(REPO / info["workspace"]), gate.load_poses(GREEN / "demo_poses.npz"), info


def test_gate_allows_the_recorded_hold_and_refuses_intruding_box_poses():
    gate, ko, poses, info = _green_gate()
    ok, d = gate.check(ko, GREEN_HOLD, poses)
    assert ok and d["envelope_voxels_inside"] == 0, d
    ok, d = gate.check(ko, BOX_TAKE_74_END, poses)
    assert not ok and d["envelope_voxels_inside"] > 0, d
    ok, d = gate.check(ko, RED_PEN_HOLD, poses)
    assert not ok, d
    g = info["other_arm_gate"]
    assert g["reason"].startswith("the box-skill end pose varies") and g["max_below"] == 0.01
    assert g["joint_lo"][4] < 29.93 < g["joint_hi"][4]  # the loose range keeps the recorded hold


def test_eval_skill_passes_the_gate(tmp_path):
    green = tmp_path / "green"
    green.mkdir()
    (green / "skill.json").write_text(json.dumps({
        "skill": "green", "arm": "left", "task": "t", "workspace": "w", "layouts": "l", "validation": {"ok": True},
        "other_arm_gate": {"joint_lo": [-5, -40, -30, -30, 20, 0], "joint_hi": [30, 35, 55, 50, 95, 100],
                           "demo_poses": "d.npz", "max_below": 0.01}}))  # fmt: skip
    out = _dry_run(tmp_path, ("SKILL", "green"))
    assert out.returncode == 0, out.stderr
    cmd = [line for line in out.stdout.splitlines() if line.startswith("python ")][-1]
    assert "--other-arm-lo -5 -40 -30 -30 20 0 --other-arm-hi 30 35 55 50 95 100 --other-arm-tol 0" in cmd
    assert "--other-arm-demo-poses d.npz --other-arm-gate-max-below 0.01" in cmd
