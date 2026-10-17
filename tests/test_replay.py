import asyncio
import json

import pytest
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


def test_repeated_evaluation_reports_false_positives_unknowns_and_agreement(tmp_path):
    images = {}
    for name in ("closed", "open", "ambiguous"):
        path = tmp_path / f"{name}.frame"
        path.write_bytes(name.encode())
        images[name] = path
    manifest = tmp_path / "labels.yaml"
    manifest.write_text(yaml.safe_dump({
        "name": "lid_states",
        "request": {
            "predicates": ["lid_open"],
            "cameras": ["top"],
            "definitions": {"lid_open": "The box lid is visibly raised."},
        },
        "frames": [
            {"name": "closed", "images": {"top": images["closed"].name},
             "expected": {"lid_open": False}},
            {"name": "open", "images": {"top": images["open"].name},
             "expected": {"lid_open": True}},
            {"name": "ambiguous", "images": {"top": images["ambiguous"].name},
             "expected": {"lid_open": "unknown"}},
        ],
    }))

    def response(value, evidence):
        return {"predicates": {"lid_open": {"value": value, "evidence": evidence}}}

    model = FixtureVisionModel([
        response("false", "lid is down"),
        response("false", "lid is down"),
        response("true", "incorrectly appears raised"),
        response("true", "interior is visible"),
        response("true", "interior is visible"),
        response("true", "interior is visible"),
        response("unknown", "view is occluded"),
        response("unknown", "view is occluded"),
        response("unknown", "view is occluded"),
    ])

    report = asyncio.run(evaluate_recording(
        load_recording(manifest),
        model,
        image_loader=lambda path: path.read_bytes(),
        repeats=3,
    ))
    payload = report.to_dict()

    assert not report.passed
    assert (report.correct, report.incorrect, report.unknown) == (8, 1, 3)
    assert report.false_positive_transitions == 1
    assert report.agreement_rate == pytest.approx(2 / 3)
    assert report.per_predicate["lid_open"].agreement_rate == pytest.approx(2 / 3)
    assert len(report.calls) == 9 and report.latency_summary["calls"] == 9
    assert report.mismatches[0].repeat == 3
    assert payload["calls"][2]["raw_response"] is None
    assert payload["calls"][2]["predicates"]["lid_open"]["false_positive_transition"]
    json.dumps(payload)


def test_repeated_evaluation_rejects_invalid_repeat_count(tmp_path):
    frame = tmp_path / "frame"
    frame.write_bytes(b"frame")
    manifest = tmp_path / "labels.yaml"
    manifest.write_text(yaml.safe_dump({
        "request": {"predicates": ["visible"], "cameras": ["top"]},
        "frames": [{"images": {"top": frame.name}, "expected": {"visible": True}}],
    }))

    with pytest.raises(ValueError, match="repeats must be a positive integer"):
        asyncio.run(evaluate_recording(
            load_recording(manifest),
            FixtureVisionModel([{"predicates": {}}]),
            image_loader=lambda path: path.read_bytes(),
            repeats=0,
        ))
