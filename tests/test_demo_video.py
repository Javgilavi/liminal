"""V4 demo-video adapter: frames, two-stage extraction, binding, corrections, end-to-end replay. Mock only."""

import asyncio
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from liminal.app import build_context
from liminal.demo import (
    DemoError,
    ReplayDemoModel,
    apply_corrections,
    bind_role,
    ground_demo,
    ground_demo_path,
    load_demo,
    synth,
)
from liminal.orchestration.episode import run_authored
from liminal.orchestration.procedure import (
    ProcedureError,
    SkillWorld,
    compile_procedure,
    scenario_with_procedure,
)
from liminal.orchestration.replay import ReplayBackend, ReplayObserver, ReplayScene, load_scenario

DEMOS = Path("configs/demos")
TWO_BLOCKS_DEMO = DEMOS / "two_blocks_demo"
STACK_DEMO = DEMOS / "stack_and_hidden_demo"
TWO_BLOCKS = "configs/replay/two_blocks.yaml"
WORKBENCH = "configs/replay/workbench.yaml"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def world_for(path):
    s = load_scenario(path)
    return SkillWorld(s.objects, s.registry)


def ground(demo_dir, scenario, **kw):
    demo = load_demo(demo_dir, **kw)
    world = world_for(scenario)
    return demo, world, *ground_demo(demo, world, ReplayDemoModel(demo))


def many_frames_demo(tmp_path, n=20):
    folder = tmp_path / "long_demo"
    (folder / "frames").mkdir(parents=True)
    for i in range(n):
        (folder / "frames" / f"f{i:03d}.png").write_bytes(synth.two_blocks_frames()[i % 6].png())
    (folder / "demo.yaml").write_text("name: long_demo\nfps: 2\n")
    return folder


# -- frames ------------------------------------------------------------------------------


def test_shipped_frames_are_small_and_deterministic(tmp_path):
    for png in DEMOS.glob("*/frames/*.png"):
        assert png.stat().st_size < 50_000 and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    a, b = synth.write_all(tmp_path / "a"), synth.write_all(tmp_path / "b")
    assert [p.read_bytes() for p in a] == [p.read_bytes() for p in b]


def test_load_demo_gives_stable_ids_and_timestamps():
    demo = load_demo(TWO_BLOCKS_DEMO)
    assert demo.name == "two_blocks_demo" and demo.source == "frames" and demo.strategy == "sampled"
    assert demo.frame_ids()[:2] == ("demo:two_blocks_demo:f000", "demo:two_blocks_demo:f001")
    assert [f.t_s for f in demo.frames][:3] == [0.0, 1.0, 2.0]
    assert "red block" in demo.transcript and demo.scenario.endswith("two_blocks.yaml")
    assert demo.resolve_frame("f002") == "demo:two_blocks_demo:f002"
    with pytest.raises(DemoError):
        demo.resolve_frame("f099")


def test_sampling_is_even_deterministic_and_keeps_first_and_last(tmp_path):
    folder = many_frames_demo(tmp_path)
    demo = load_demo(folder, max_frames=5)
    assert [f.index for f in demo.frames] == [0, 5, 10, 14, 19]
    assert [f.index for f in load_demo(folder, max_frames=5).frames] == [f.index for f in demo.frames]
    assert len(load_demo(folder, max_frames=100).frames) == 20
    assert [f.t_s for f in demo.frames][1] == 2.5  # fps: 2


def test_manual_vs_sampled_label_is_recorded_and_surfaced():
    sampled = load_demo(TWO_BLOCKS_DEMO)
    manual = load_demo(TWO_BLOCKS_DEMO, strategy="manual")
    assert sampled.strategy == "sampled" and len(sampled.frames) == 6
    assert manual.strategy == "manual" and [f.index for f in manual.frames] == [0, 2, 4, 5]
    _, _, _, extraction = ground(TWO_BLOCKS_DEMO, TWO_BLOCKS, strategy="manual")
    assert extraction.to_dict()["strategy"] == "manual"
    with pytest.raises(DemoError, match="keyframes.json"):
        load_demo(STACK_DEMO, strategy="manual")
    with pytest.raises(DemoError, match="strategy"):
        load_demo(TWO_BLOCKS_DEMO, strategy="magic")


def test_manual_keyframes_accept_seconds_and_reject_unknown(tmp_path):
    folder = many_frames_demo(tmp_path)
    (folder / "keyframes.json").write_text(json.dumps({"frames": [0, 2.5, "f019"]}))
    assert [f.index for f in load_demo(folder, strategy="manual").frames] == [0, 5, 19]
    (folder / "keyframes.json").write_text(json.dumps(["f777"]))
    with pytest.raises(DemoError, match="no frame"):
        load_demo(folder, strategy="manual")


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_video_is_decoded_with_ffmpeg_and_frames_convert_to_jpeg(tmp_path):
    folder = tmp_path / "vid_demo"
    folder.mkdir()
    made = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=3",
                           "-c:v", "mpeg4", str(folder / "video.mp4")], capture_output=True)
    if made.returncode != 0:
        pytest.skip("this ffmpeg cannot encode a test video")
    (folder / "demo.yaml").write_text("name: vid_demo\nfps: 2\n")
    demo = load_demo(folder, max_frames=4)
    assert demo.source == "video" and len(demo.frames) == 4 and demo.frames[0].media_type == "image/jpeg"
    assert demo.frames[-1].index == 5 and demo.frames[-1].frame_id == "demo:vid_demo:f005"
    (folder / "keyframes.json").write_text("[0, 1.5]")
    assert [f.index for f in load_demo(folder, strategy="manual").frames] == [0, 3]
    png = load_demo(TWO_BLOCKS_DEMO)
    assert png.model_images()[0][:2] == b"\xff\xd8"  # PNG fixtures are sent to the model as JPEG


def test_empty_or_missing_demo_folders_fail_loudly(tmp_path):
    with pytest.raises(DemoError, match="not found"):
        load_demo(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(DemoError, match="neither"):
        load_demo(tmp_path / "empty")


# -- extraction ----------------------------------------------------------------------------


def test_extraction_gives_a_validated_proposal_that_cites_frames():
    demo, world, proposal, extraction = ground(TWO_BLOCKS_DEMO, TWO_BLOCKS)
    assert proposal.source_type == "demo_video"
    assert proposal.obligations == ("red_in_box", "blue_in_box")
    assert [(s.step_id, s.object_id, s.destination_id) for s in proposal.steps] == [
        ("d1", "red", "box"), ("d2", "blue", "box")]
    assert proposal.steps[1].depends_on == ("d1",)
    for s in proposal.steps:
        frame_refs = [r.ref for r in s.sources if r.kind == "demo_frame"]
        assert frame_refs and set(frame_refs) <= set(demo.frame_ids())
        assert s.confidence == 0.95
        assert any(r.kind == "demo_transcript" for r in s.sources)
    assert (proposal.unresolved, proposal.unsupported) == ((), ())
    assert [c.kind for c in extraction.constraints] == ["order"]
    assert extraction.to_dict()["subtasks"][0]["frames"][0] == "demo:two_blocks_demo:f000"


def test_scripted_model_is_labelled_scripted_and_never_a_hardware_result():
    _, _, _, extraction = ground(TWO_BLOCKS_DEMO, TWO_BLOCKS)
    d = extraction.to_dict()
    assert d["scripted"] is True and "scripted" in d["model"] and "annotation" in d["model"]
    assert "not a hardware run" in d["evidence"]


def test_unsupported_hidden_and_uncertain_actions_are_reported_not_mapped():
    _, _, proposal, extraction = ground(STACK_DEMO, WORKBENCH)
    assert [s.step_id for s in proposal.steps] == ["d1", "d4"]
    assert proposal.obligations == ("green_in_left_bin", "blue_in_box")
    assert len(proposal.unsupported) == 1 and proposal.unsupported[0].startswith("d2:")
    text = "\n".join(proposal.unresolved)
    assert "d3: hidden action not mapped" in text and "d4: demonstrator action is uncertain" in text
    assert "d4: follows d2" in text  # order constraint on a step that was not mapped
    d4 = proposal.steps[1]
    assert d4.confidence == 0.55 and d4.confidence <= 0.6 and proposal.steps[0].confidence == 0.95
    assert [s.hidden for s in extraction.subtasks] == [False, False, True, False]


def test_uncertain_confidence_is_capped_even_if_the_model_overclaims():
    class Overconfident(ReplayDemoModel):
        def _map(self, request):
            out = super()._map(request)
            for s in out["steps"]:
                s["confidence"] = 0.99
            return out

    demo = load_demo(STACK_DEMO)
    proposal, _ = ground_demo(demo, world_for(WORKBENCH), Overconfident(demo))
    assert {s.step_id: s.confidence for s in proposal.steps} == {"d1": 0.99, "d4": 0.6}


# -- changed layout and role binding ---------------------------------------------------------


def test_demo_replays_in_a_changed_layout_by_label():
    base = world_for(WORKBENCH)
    renamed = SkillWorld({"r1": "red block", "b1": "blue block", "tray": "white box", "g1": "green block"},
                         base.registry)
    demo = load_demo(TWO_BLOCKS_DEMO)
    for world, ids in [(base, ("red", "blue", "box")), (renamed, ("r1", "b1", "tray"))]:
        proposal, _ = ground_demo(demo, world, ReplayDemoModel(demo))
        assert [(s.object_id, s.destination_id) for s in proposal.steps] == [(ids[0], ids[2]), (ids[1], ids[2])]


def test_a_role_that_cannot_be_bound_is_reported_never_guessed():
    # the stack demo needs a green block and a left bin; two_blocks has neither
    demo, _, proposal, extraction = ground(STACK_DEMO, TWO_BLOCKS)
    assert [s.step_id for s in proposal.steps] == ["d4"]
    assert any(m.startswith("d1: cannot bind role") and "green block" in m for m in proposal.unresolved)
    green = extraction.bindings["green block"]
    assert green.object_id is None and "green block" in green.reason
    assert "green" not in {s.object_id for s in proposal.steps}


def test_bind_role_by_label_and_never_ambiguously():
    objects = world_for(WORKBENCH).objects
    assert bind_role("the red block", objects).object_id == "red"
    assert bind_role("white box", objects).object_id == "box"
    assert bind_role("the bin on the left", objects).object_id == "left_bin"
    assert bind_role("a block", objects).object_id is None and "ambiguous" in bind_role("a block", objects).reason
    assert bind_role("bin", objects).object_id is None
    assert bind_role("purple cube", objects).object_id is None
    assert bind_role("the", objects).object_id is None


def test_model_guessing_an_object_for_an_unbound_or_mismatched_role_is_repaired_then_rejected():
    demo = load_demo(TWO_BLOCKS_DEMO)
    world = world_for(TWO_BLOCKS)

    class Guesser(ReplayDemoModel):
        def _map(self, request):
            out = super()._map(request)
            out["steps"][0]["object_id"] = "blue"  # d1 says "red block"
            return out

    with pytest.raises(DemoError, match="object_id must be 'red'"):
        ground_demo(demo, world, Guesser(demo))


def test_model_citing_a_frame_that_was_not_shown_is_rejected():
    demo = load_demo(TWO_BLOCKS_DEMO, strategy="manual")  # f001 and f003 are not loaded

    class Cheater(ReplayDemoModel):
        def _map(self, request):
            out = super()._map(request)
            out["steps"][0]["sources"].append({"kind": "demo_frame", "ref": "demo:two_blocks_demo:f001"})
            return out

    with pytest.raises(DemoError, match="not one of the loaded frames"):
        ground_demo(demo, world_for(TWO_BLOCKS), Cheater(demo))
    # the scripted stage 1 cites the nearest LOADED frame instead of an unloaded one
    proposal, _ = ground_demo(demo, world_for(TWO_BLOCKS), ReplayDemoModel(demo))
    assert {r.ref for s in proposal.steps for r in s.sources if r.kind == "demo_frame"} <= set(demo.frame_ids())


def test_invented_skill_or_stray_field_is_repaired_once_then_fails():
    demo = load_demo(TWO_BLOCKS_DEMO)
    world = world_for(TWO_BLOCKS)

    class Bad(ReplayDemoModel):
        map_calls = 0

        def _map(self, request):
            Bad.map_calls += 1
            out = super()._map(request)
            out["steps"][0]["skill"] = "teleport"
            return out

    with pytest.raises(DemoError, match="unregistered skill"):
        ground_demo(demo, world, Bad(demo))
    assert Bad.map_calls == 2  # one first attempt + one repair round

    class Once(ReplayDemoModel):
        first = True

        def _map(self, request):
            out = super()._map(request)
            if Once.first:
                Once.first = False
                out["steps"][0]["code"] = "import os"
            return out

    proposal, extraction = ground_demo(demo, world, Once(demo))
    assert len(proposal.steps) == 2 and extraction.repairs == 1


def test_support_constraint_becomes_maintain_and_blocks_compilation(tmp_path):
    folder = tmp_path / "support_demo"
    (folder / "frames").mkdir(parents=True)
    for i in range(3):
        (folder / "frames" / f"f{i:03d}.png").write_bytes(synth.two_blocks_frames()[i].png())
    (folder / "demo.yaml").write_text("name: support_demo\n")
    ann = yaml.safe_load((TWO_BLOCKS_DEMO / "annotation.yaml").read_text())
    ann["constraints"] = [{"kind": "support", "before": "d1", "after": "d2", "condition": "red_in_box_stays"}]
    (folder / "annotation.yaml").write_text(yaml.safe_dump(ann))
    demo = load_demo(folder)
    world = world_for(TWO_BLOCKS)
    proposal, _ = ground_demo(demo, world, ReplayDemoModel(demo))
    assert proposal.steps[1].maintain == ("red_in_box_stays",) and proposal.steps[1].depends_on == ("d1",)
    with pytest.raises(ProcedureError, match="maintained"):
        compile_procedure(proposal, world)


# -- corrections -----------------------------------------------------------------------------


def test_corrections_revalidate_and_count():
    _, world, proposal, _ = ground(TWO_BLOCKS_DEMO, WORKBENCH)
    fixed = apply_corrections(proposal, [
        {"op": "arm", "step": "d1", "arm": "right"},
        {"op": "destination", "step": "d2", "destination_id": "right_bin"},
    ], world)
    assert fixed.corrections == 2 and fixed.first_pass is proposal
    assert fixed.proposal.steps[0].arm == "right"
    assert fixed.proposal.steps[1].effect == "blue_in_right_bin"
    assert fixed.proposal.obligations == ("red_in_box", "blue_in_right_bin")
    assert any(r.kind == "operator" for r in fixed.proposal.steps[1].sources)
    assert proposal.steps[1].effect == "blue_in_box"  # the first pass is untouched
    assert apply_corrections(proposal, [], world).corrections == 0
    assert fixed.to_dict()["corrections"] == 2


def test_remove_and_reorder_corrections():
    _, world, proposal, _ = ground(TWO_BLOCKS_DEMO, WORKBENCH)
    removed = apply_corrections(proposal, [{"op": "remove", "step": "d1"}], world)
    assert [s.step_id for s in removed.proposal.steps] == ["d2"] and removed.proposal.steps[0].depends_on == ()
    assert removed.proposal.obligations == ("blue_in_box",) and removed.corrections == 1
    # d2 depends on d1, so swapping them is a contradiction the validator must refuse
    with pytest.raises(ProcedureError, match="depends_on"):
        apply_corrections(proposal, [{"op": "reorder", "order": ["d2", "d1"]}], world)
    free = replace(proposal, steps=tuple(replace(s, depends_on=()) for s in proposal.steps))
    swapped = apply_corrections(free, [{"op": "reorder", "order": ["d2", "d1"]}], world)
    assert [s.step_id for s in swapped.proposal.steps] == ["d2", "d1"]
    assert swapped.proposal.obligations == ("blue_in_box", "red_in_box")


@pytest.mark.parametrize("bad, match", [
    ({"op": "remove", "step": "d9"}, "unknown step"),
    ({"op": "explode", "step": "d1"}, "op must be one of"),
    ({"op": "arm", "step": "d1", "arm": "third"}, "cannot run"),
    ({"op": "destination", "step": "d1", "destination_id": "unicorn"}, "unknown destination"),
    ({"op": "reorder", "order": ["d1"]}, "every current step"),
])
def test_bad_corrections_are_rejected(bad, match):
    _, world, proposal, _ = ground(TWO_BLOCKS_DEMO, WORKBENCH)
    with pytest.raises(ProcedureError, match=match):
        apply_corrections(proposal, [bad], world)


# -- end to end ------------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", [TWO_BLOCKS, WORKBENCH])
def test_demo_runs_end_to_end_on_replay_with_fresh_verification(scenario, tmp_path):
    demo, world, proposal, _ = ground(TWO_BLOCKS_DEMO, scenario)
    task = compile_procedure(proposal, world, name="demo_two_blocks")
    scenario_obj = scenario_with_procedure(load_scenario(scenario), proposal)
    scene = ReplayScene(scenario_obj, tmp_path / "scene.json")
    ctx = build_context("configs/robot/mock_dual.yaml", backend="mock",
                        policy_backend=ReplayBackend(scene, lie_first=True))
    observer = ReplayObserver(ctx.world, scene, ttl_s=30.0, bound=False)
    report = asyncio.run(run_authored(scenario_obj, ctx, task, observer))
    assert report.completed, report.summary()
    assert {o.obligation_id for o in report.obligations} == {"red_in_box", "blue_in_box"}
    assert [a.status for a in report.attempts].count("verified") == 2  # the first (lying) attempt did not count


def test_unsupported_demo_still_runs_its_supported_part_on_the_workbench(tmp_path):
    _, world, proposal, _ = ground(STACK_DEMO, WORKBENCH)
    scenario = scenario_with_procedure(load_scenario(WORKBENCH), proposal)
    scene = ReplayScene(scenario, tmp_path / "scene.json")
    ctx = build_context("configs/robot/mock_dual.yaml", backend="mock", policy_backend=ReplayBackend(scene))
    observer = ReplayObserver(ctx.world, scene, ttl_s=30.0, bound=False)
    report = asyncio.run(run_authored(scenario, ctx, compile_procedure(proposal, world), observer))
    assert report.completed and {o.obligation_id for o in report.obligations} == {"green_in_left_bin", "blue_in_box"}


def test_service_helper_returns_proposal_rows_and_labelled_extraction():
    proposal, rows, extraction = ground_demo_path(WORKBENCH, str(TWO_BLOCKS_DEMO))
    assert proposal.source_type == "demo_video" and [r["id"] for r in rows] == ["d1", "d2"]
    assert rows[0]["flux_prompt"] == "Pick up the red block and place it inside the white box."
    assert rows[0]["sources"][0]["kind"] == "demo_frame" and extraction["scripted"] is True
    with pytest.raises(ValueError, match="unknown model"):
        ground_demo_path(WORKBENCH, str(TWO_BLOCKS_DEMO), "gpt")
