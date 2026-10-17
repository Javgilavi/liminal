import asyncio

import yaml

from liminal.perception import FixtureVisionModel, evaluate_recording, load_recording


def test_recorded_frames_run_through_production_inspector(tmp_path):
    first = tmp_path / "closed.frame"
    second = tmp_path / "open.frame"
    first.write_bytes(b"closed")
    second.write_bytes(b"open")
    manifest = tmp_path / "box.yaml"
    manifest.write_text(yaml.safe_dump({
        "name": "box_states",
        "request": {
            "predicates": ["lid_open"],
            "cameras": ["top"],
            "definitions": {"lid_open": "The lid is raised and the box interior is visible."},
        },
        "frames": [
            {"name": "closed", "images": {"top": first.name}, "expected": {"lid_open": False}},
            {"name": "open", "images": {"top": second.name}, "expected": {"lid_open": True}},
        ],
    }))
    model = FixtureVisionModel([
        {"predicates": {"lid_open": {"value": "false", "evidence": "lid is down"}}},
        {"predicates": {"lid_open": {"value": "true", "evidence": "interior is visible"}}},
    ])

    report = asyncio.run(evaluate_recording(
        load_recording(manifest),
        model,
        image_loader=lambda path: path.read_bytes(),
    ))

    assert report.passed and report.accuracy == 1.0
    assert report.frames == 2 and report.observations == 2 and report.unknown == 0
    assert [images for _, images in model.calls] == [[b"closed"], [b"open"]]
