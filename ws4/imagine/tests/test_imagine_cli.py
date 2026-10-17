import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")

from ws4.imagine.run_imagine import build_parser, main  # noqa: E402


def test_fake_run_writes_gif_sheet_and_report(tmp_path):
    report = main(["fake", "--out", str(tmp_path), "--steps", "40"])
    assert report["steps"] == [0, 32] and report["frames_per_chunk"] == 41
    assert (tmp_path / "side_by_side.gif").exists() and (tmp_path / "contact_sheet.png").exists()
    assert json.loads((tmp_path / "report.json").read_text())["chunks"] == 2


def test_gpu_commands_parse_without_gpu():
    args = build_parser().parse_args(
        ["replay", "--dataset", "lerobot/x", "--scene-key", "a", "--wrist-key", "b", "--decode-device", "cuda:1"]
    )
    assert args.policy == "black-forest-labs/flux-3-action-so101" and args.decode_device == "cuda:1"
    assert build_parser().parse_args(["bench"]).candidates == [2, 4]


def test_nearest_sources_finds_matching_frames():
    import numpy as np

    from ws4.imagine.run_imagine import nearest_sources

    source = np.stack([np.full((4, 8, 3), v, np.uint8) for v in (0, 50, 100, 150)])
    assert nearest_sources(source[[1, 3, 2]] + 3, source) == [1, 3, 2]
    assert build_parser().parse_args(["vaecheck", "--dataset", "d", "--scene-key", "a", "--wrist-key", "b"]).start == 8
