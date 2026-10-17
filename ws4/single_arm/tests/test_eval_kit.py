"""Post-training eval kit: layout sequence, placement references and page, start/parity checks, trial fields."""

import json
import sys
import urllib.request
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

cv2 = pytest.importorskip("cv2")

import placement  # noqa: E402
import preflight  # noqa: E402
import run_single_arm as rsa  # noqa: E402

LAYOUTS = HERE / "layouts"


def shifted(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def test_attempt_plan_runs_each_layout_in_a_block():
    plan = rsa.attempt_plan(Namespace(layouts=["A", "B", "C", "D"], per_layout=3, layout="", attempts=1))
    assert plan == [x for x in "ABCD" for _ in range(3)]
    assert rsa.attempt_plan(Namespace(layouts=None, per_layout=3, layout="cup1", attempts=2)) == ["cup1", "cup1"]


def test_layout_references_are_the_held_out_takes():
    refs = placement.load_refs(LAYOUTS)
    split = {"A": [20, 21, 22], "B": [36, 37, 38], "C": [86, 87, 88], "D": [101, 102, 103]}
    assert {k: [int(p.stem.split("_ep")[1][:3]) for p in v] for k, v in refs.items()} == split
    index = json.loads((LAYOUTS / "index.json").read_text())
    assert index["task"] == "put the red pen in its slot in the cardboard box"
    for take in index["takes"]:  # every held-out take starts at haller home, like the eval
        assert np.allclose(take["start_joints_task_arm"][:5], 0, atol=3.5)


def test_composite_marks_reference_edges_on_the_live_view():
    ref = placement.read_rgb(LAYOUTS / "A_ep020_head.jpg")
    out = placement.composite(ref, ref, "x")
    h, w = ref.shape[:2]
    assert out.shape == (2 * h, 2 * w, 3)
    green = (out[:h, w:] == (0, 255, 0)).all(axis=-1)
    assert 0.01 < green.mean() < 0.4


def test_base_plate_shift_ignores_the_scene_above_it():
    ref = placement.read_rgb(LAYOUTS / "A_ep020_head.jpg")
    other_layout = placement.read_rgb(LAYOUTS / "D_ep101_head.jpg")  # box and pen elsewhere, same camera
    dx, dy, rot = placement.ref_shift(other_layout, ref)
    assert abs(dx) < 4 and abs(dy) < 4 and abs(rot) < 0.5, (dx, dy, rot)
    dx, dy, _ = placement.ref_shift(shifted(ref, 9, -6), ref)
    assert abs(dx - 9) < 1.5 and abs(dy + 6) < 1.5, (dx, dy)


def test_parity_flags_a_rotated_wrist_and_a_moved_head_camera():
    wrist = placement.read_rgb(LAYOUTS / "A_ep020_wrist.jpg")
    head = placement.read_rgb(LAYOUTS / "A_ep020_head.jpg")
    assert preflight.parity(placement.read_rgb(LAYOUTS / "B_ep036_wrist.jpg"), wrist, "wrist")[0]
    assert not preflight.parity(np.rot90(wrist, 2).copy(), wrist, "wrist")[0]
    assert preflight.parity(placement.read_rgb(LAYOUTS / "C_ep086_head.jpg"), head, "head")[0]
    assert not preflight.parity(shifted(head, 25, 0), head, "head")[0]
    assert not preflight.parity((head * 0.5).astype(np.uint8), head, "head")[0]  # much darker


def test_placement_page_serves_the_latest_composite():
    view = placement.PlacementView(port=0)
    try:
        port = view.server.server_address[1]
        ref = placement.read_rgb(LAYOUTS / "A_ep020_head.jpg")
        view.update(ref, ref, "test")
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read()
        jpg = urllib.request.urlopen(f"http://127.0.0.1:{port}/frame.jpg?t=1").read()
        assert b"frame.jpg" in page and jpg[:2] == b"\xff\xd8"
    finally:
        view.close()


def test_logged_trial_carries_condition_layout_and_checkpoint(tmp_path, monkeypatch):
    log = tmp_path / "trials.jsonl"
    monkeypatch.setenv("WS4_TRIAL_LOG", str(log))
    monkeypatch.setattr(rsa, "STEPS", rsa.TASK_STEPS["put_pen_in_slot"])
    args = Namespace(trial_task="put_pen_in_slot", condition="post_training", variant="lora_5000", layout="",
                     note="", guidance_note="1.0", max_step=1.0, max_relative_target=20.0, torque_limit=60.0,
                     operator="oscar")  # fmt: skip
    meta = {"checkpoint": "pkg_lora5000", "layout": "C", "layout_ref": "C_ep086_head", "min_supply_v": 7.1,
            "keepout_refusals": {}, "end_reason": "timeout", "chunks": 3, "guard_ticks": 0}  # fmt: skip
    rsa.log_trial(args, "put the red pen in its slot in the cardboard box", meta,
                  {"steps_done": 2, "interventions": 0}, tmp_path)  # fmt: skip
    row = json.loads(log.read_text())
    assert (row["condition"], row["layout"], row["checkpoint"], row["steps_total"]) == \
        ("post_training", "C", "pkg_lora5000", 3)  # fmt: skip
    assert row["failure"] == "stopped before: insert it in its slot" and "C_ep086_head" in row["notes"]


def test_home_check_passes_recorded_home_frames_and_flags_a_moved_arm():
    import home_check as hc

    ref_dir = HERE / "frames" / "reference_2026-10-17_night"
    if not ref_dir.exists():
        pytest.skip("ws4-vr2's reference frames are not in this checkout")
    rh, rw, state, name = hc.load_refs(ref_dir)
    assert name.startswith("ep103") and np.allclose(state[:5], [0, 2.5, 2.5, 0.5, 0])
    home = np.array([0, 2.0, 1.5, 0.3, 0, 99.0])
    for take in ("A_ep020", "C_ep086"):  # other layouts, same calibration, arm at home
        head, wrist = (placement.read_rgb(LAYOUTS / f"{take}_{c}.jpg") for c in ("head", "wrist"))
        ok, lines, img = hc.home_check(head, wrist, home, rh, rw, state)
        assert ok, lines
        assert img.shape == (2 * rh.shape[0], 3 * rh.shape[1], 3)
    ok, lines, _ = hc.home_check(shifted(rh, 25, 15), rw, home, rh, rw, state)  # arm drawn elsewhere
    assert not ok and any("NOT where" in line for line in lines)
    ok, lines, _ = hc.home_check(rh, rw, home + [0, 9, 0, 0, 0, 0], rh, rw, state)  # lift 9 deg off home
    assert not ok and any("max error" in line for line in lines)
