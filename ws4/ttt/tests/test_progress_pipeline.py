"""Labels, features, the GPU request builder (on CPU), training and the orchestrator API, on a fake episode."""

import io
import json

import numpy as np
import pytest
import torch
from ttt.features import FrameFeaturizer, episode_sequence, plan_occurrences, stage_posterior
from ttt.labels import EpisodeLabels, Segment, replan_targets, segments_from_events

PLAN = [("open_lid", "left"), ("close_lid", "left"), ("open_lid", "left"), ("close_lid", "left")]


def _jpeg(img):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=90)
    return np.frombuffer(buf.getvalue(), np.uint8)


def make_episode(root, name="lid_cycles_2-000001", seed=0, seg_len=96):
    """A fake episode in the sim_episodes.py layout: 4 steps of `seg_len` ticks, a re-plan every 32 ticks."""
    rng = np.random.default_rng(seed)
    d = root / name
    d.mkdir(parents=True)
    steps = [{"id": f"s{i + 1}", "skill": s, "arm": a} for i, (s, a) in enumerate(PLAN)]
    segments, replans, frame_ticks = [], [], set()
    t = 0
    for st in steps:
        start = t
        for c in range(seg_len // 32):
            tick = start + 32 * c
            snaps = [max(start, tick - 7), tick]
            replans.append(
                {"tick": tick, "arm": st["arm"], "step_id": st["id"], "prompt": st["skill"], "snapshot_ticks": snaps}
            )
            frame_ticks.update(snaps)
        t = start + seg_len
        segments.append(Segment(st["id"], st["skill"], st["arm"], start, t, True, 1))
    n = t + 4
    labels = EpisodeLabels(
        name,
        "sim_gt",
        30.0,
        n,
        "lid_cycles_2",
        steps,
        segments,
        [r["tick"] for r in replans],
        extra={"replans": replans},
    )
    state = rng.normal(0, 20, (n, 12)).astype(np.float32)
    np.savez_compressed(d / "ticks.npz", state=state, command=state + 1, stage=np.zeros(n, np.int64))
    ft = sorted(frame_ticks)
    frames = {"frame_ticks": np.asarray(ft)}
    for cam in ("head", "left_wrist", "right_wrist"):
        frames[cam] = np.array([_jpeg(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)) for _ in ft], dtype=object)
    np.savez(d / "frames.npz", **frames)
    labels.save(d / "episode.json")
    return d


def test_labels_roundtrip_targets_and_aliasing(tmp_path):
    d = make_episode(tmp_path)
    lab = EpisodeLabels.load(d / "episode.json")
    tg = replan_targets(lab)
    assert list(tg.stage) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert list(tg.done) == [0, 0, 1] * 4
    assert list(tg.occurrence) == [0] * 6 + [1] * 6  # the second open/close cycle repeats both skills
    np.testing.assert_allclose(tg.progress[3], 0.25)
    with pytest.raises(ValueError):
        EpisodeLabels("x", "nope", 30, 10, "t", [{"id": "a", "skill": "s", "arm": "left"}], [])


def test_segments_overlap_allowed_across_arms_only():
    steps = [{"id": "a", "skill": "s", "arm": "left"}, {"id": "b", "skill": "t", "arm": "right"}]
    EpisodeLabels(
        "x", "manual", 30, 100, "t", steps, [Segment("a", "s", "left", 0, 50), Segment("b", "t", "right", 10, 60)]
    )
    with pytest.raises(ValueError):
        steps2 = [{"id": "a", "skill": "s", "arm": "left"}, {"id": "b", "skill": "t", "arm": "left"}]
        EpisodeLabels(
            "x", "manual", 30, 100, "t", steps2, [Segment("a", "s", "left", 0, 50), Segment("b", "t", "left", 10, 60)]
        )


def test_orchestrator_event_log_to_segments():
    events = [
        {"t": 0.5, "step_id": "s1", "skill": "open_lid", "arm": "left", "event": "start"},
        {"t": 9.0, "step_id": "s1", "event": "end", "success": False},
        {"t": 9.5, "step_id": "s1", "skill": "open_lid", "arm": "left", "event": "start"},
        {"tick": 600, "step_id": "s1", "event": "end", "success": True},
    ]
    segs = segments_from_events(events, fps=30)
    assert [(s.start_tick, s.end_tick, s.attempt, s.success) for s in segs] == [
        (15, 270, 1, False),
        (285, 600, 2, True),
    ]


def test_stage_posterior_uses_plan_and_occurrence():
    skills = ["close_lid", "open_lid"]
    plan = ["open_lid", "close_lid", "open_lid", "close_lid"]
    assert plan_occurrences(plan) == [0, 0, 1, 1]
    skill_p = np.array([0.1, 0.9])
    occ_p = np.array([0.2, 0.8, 0, 0, 0, 0])
    post = stage_posterior(skill_p, occ_p, plan, skills)
    assert post.argmax() == 2 and abs(post.sum() - 1) < 1e-9


def test_frame_features_and_sequence(tmp_path):
    d = make_episode(tmp_path)
    seq = episode_sequence(d, ["close_lid", "open_lid"], "frame", FrameFeaturizer())
    assert seq.feat.shape == (12, FrameFeaturizer().dim)
    assert (d / "features_frame.npz").exists()
    assert list(seq.skill[:4]) == [1, 1, 1, 0]


def test_extraction_requests_match_the_runner_contract(tmp_path):
    pytest.importorskip("flux.contract")
    from flux.backends import FakeBackend, validate_request
    from flux.contract import SO101_CONTRACT
    from ttt.extract_flux_features import episode_requests

    d = make_episode(tmp_path)
    reqs = list(episode_requests(d, SO101_CONTRACT))
    assert len(reqs) == 12
    rp, req = reqs[1]  # second re-plan of step 1: window reaches back 7 ticks, all inside the step
    validate_request(req, SO101_CONTRACT)
    z = np.load(d / "ticks.npz")
    np.testing.assert_allclose(req.states[-1], z["state"][32][:6])
    np.testing.assert_allclose(req.commands[-1], z["command"][31][:6])
    _, first = reqs[0]  # first re-plan: the step start is repeated, the previous command is the state
    np.testing.assert_allclose(first.commands[0], z["state"][0][:6])
    assert FakeBackend(SO101_CONTRACT).predict(req).commands.shape[0] == SO101_CONTRACT.chunk_size


def test_train_save_and_progress_monitor_roundtrip(tmp_path):
    from ttt.progress_api import ProgressMonitor
    from ttt.train_progress import main as train_main

    for i in range(6):
        make_episode(tmp_path, name=f"lid_cycles_2-{i:06d}", seed=i)
    res = train_main(
        [
            "--data",
            str(tmp_path),
            "--memories",
            "ttt",
            "--steps",
            "3",
            "--save-dir",
            str(tmp_path / "m"),
            "--out",
            str(tmp_path / "r.json"),
        ]
    )
    assert "ttt" in res["models"] and json.loads((tmp_path / "r.json").read_text())["features"] == "frame"
    mon = ProgressMonitor.load(tmp_path / "m" / "progress_frame_ttt_s0.pt")
    mon.reset(plan=[s for s, _ in PLAN])
    rng = np.random.default_rng(0)
    for _ in range(3):
        frames = {
            f"observation.images.{c}": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
            for c in ("head", "left_wrist", "right_wrist")
        }
        est = mon.update(frames=frames, state=np.zeros(12))
    assert 0 <= est.stage < 4 and 0 <= est.done <= 1 and len(est.stage_probs) == 4 and len(mon.history) == 3
    with pytest.raises(ValueError):
        mon.reset(plan=["fly_to_moon"])
    torch.manual_seed(0)


def test_claude_labeller_turns_frame_answers_into_segments_without_calling_the_api(tmp_path):
    from types import SimpleNamespace

    from ttt.claude_stage_labels import evaluate, label_episode, to_labels

    d = make_episode(tmp_path)

    class FakeMessages:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            n = sum(1 for c in kwargs["messages"][0]["content"] if c["type"] == "image")
            frames = [{"frame": i, "step": min(3, i * 4 // n), "steps_done": min(3, i * 4 // n)} for i in range(n)]
            text = json.dumps({"frames": frames, "notes": ""})
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text=text)],
                usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
                model=kwargs["model"],
            )

    fake = FakeMessages()
    client = SimpleNamespace(messages=fake, beta=SimpleNamespace(messages=fake))
    result = label_episode(client, d, n_frames=8, model="claude-opus-5-5", effort="low")
    assert fake.kwargs["fallbacks"] == "default" and fake.kwargs["output_config"]["format"]["type"] == "json_schema"
    auto = to_labels(d, result)
    assert auto.source == "claude_vision" and [s.step_id for s in auto.segments] == ["s1", "s2", "s3", "s4"]
    ev = evaluate(d, result, auto)
    assert ev["frame_acc"] > 0.7 and ev["steps_found"] == "4/4"
