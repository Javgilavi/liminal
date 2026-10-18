"""The FLUX-in-sim closed loop: plan parsing, scoring, and one short episode with a still policy (no GPU)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
from ttt.flux_sim_closed_loop import monitor_scores, parse_plan, summarize
from ttt.labels import EpisodeLabels, Segment


def test_parse_plan_defaults_to_left_arm():
    assert parse_plan("open_lid:left, put_block_in_box:right,close_lid") == [
        ("open_lid", "left"),
        ("put_block_in_box", "right"),
        ("close_lid", "left"),
    ]


def _labels() -> EpisodeLabels:
    steps = [{"id": "s1", "skill": "open_lid", "arm": "left"}, {"id": "s2", "skill": "close_lid", "arm": "left"}]
    segments = [Segment("s1", "open_lid", "left", 0, 64, False, 1), Segment("s1", "open_lid", "left", 64, 128, True, 2),
                Segment("s2", "close_lid", "left", 128, 192, True, 1)]  # fmt: skip
    replans = []
    for tick, backend, est in ((0, "flux", 0), (32, "flux", 0), (64, "scripted", 1), (96, "scripted", 0),
                               (128, "flux", 1), (160, "flux", 1)):  # fmt: skip
        replans.append({"tick": tick, "backend": backend, "estimate": {"stage": est, "progress": 0.0}})
    return EpisodeLabels("e", "sim_gt", 30.0, 200, "t", steps, segments, [r["tick"] for r in replans],
                         {"replans": replans})  # fmt: skip


def test_monitor_scores_split_by_backend():
    s = monitor_scores(_labels())
    assert s["n_all"] == 6 and s["n_flux"] == 4 and s["n_scripted"] == 2
    assert s["stage_acc_flux"] == 1.0 and s["stage_acc_scripted"] == 0.5
    assert s["stage_acc_all"] == pytest.approx(5 / 6)


def test_summarize_counts_only_valid_preconditions():
    steps = [{"skill": "open_lid", "precondition_ok": True, "flux_success": False},
             {"skill": "close_lid", "precondition_ok": False, "flux_success": True}]  # fmt: skip
    out = summarize([{"steps": steps, "monitor": monitor_scores(_labels())}])
    assert out["flux_success"] == {"open_lid": {"success": 0, "steps": 1}}
    assert out["monitor"]["all"]["replans"] == 6


def _sim(render: bool = True):
    pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL", "egl")
    from ttt.flux_sim_closed_loop import _imports

    _, _, _, _, contract, _, sb, load_vocabulary = _imports()
    try:
        sim = sb.PackBoxSim(image_size=(120, 160), render=render, cameras=("head", "left_wrist", "right_wrist"))
        robot = sb.SimRobot(sim, seed=0)
    except Exception as exc:  # no EGL on this machine
        pytest.skip(f"sim rendering unavailable: {exc}")
    return sim, robot, contract, sb, load_vocabulary()


def test_still_policy_episode_is_repaired_and_loadable(tmp_path):
    from flux.backends import FakeBackend
    from ttt.features import FrameFeaturizer, frame_features
    from ttt.flux_sim_closed_loop import run_episode

    sim, robot, contract, _, vocab = _sim()
    args = SimpleNamespace(backend="fake", package="fake", url="", max_chunks=1, repair=True, monitor=None)
    ep = tmp_path / "ep"
    labels, results = run_episode(args, sim, robot, FakeBackend(contract, velocity=0.0), vocab,
                                  [("open_lid", "left")], 0, ep, monitor=None, video=False)  # fmt: skip
    assert results[0]["precondition_ok"] and not results[0]["flux_success"]
    assert results[0]["repair"]["scripted"]["success"]
    assert [s.attempt for s in labels.segments] == [1, 2]
    loaded = EpisodeLabels.load(ep / "episode.json")
    feats = frame_features(ep, FrameFeaturizer(), cache=False)
    assert feats.shape[0] == len(loaded.replan_ticks) and np.isfinite(feats).all()


def test_unplannable_stand_in_attempt_fails_cleanly_with_lazy_rendering(tmp_path):
    """--backend scripted: an IK failure on the first attempt is a failed attempt, not a crash; the sim renders
    only the snapshot ticks."""
    from ttt.flux_sim_closed_loop import run_episode

    sim, robot, _, sb, vocab = _sim(render=False)

    class Unplannable(sb.ScriptedSimBackend):
        def begin(self, *a, **k):
            raise sb.SimSkillError("unreachable")

    args = SimpleNamespace(backend="scripted", package="scripted", url="", max_chunks=2, repair=True, monitor=None,
                           render_every_tick=False)  # fmt: skip
    labels, results = run_episode(args, sim, robot, Unplannable(sim), vocab, [("open_lid", "left")], 0,
                                  tmp_path / "ep", monitor=None, video=False)  # fmt: skip
    assert results[0]["chunks"] == 0 and not results[0]["flux_success"]
    assert results[0]["repair"]["scripted"]["success"]
    frames = np.load(tmp_path / "ep" / "frames.npz", allow_pickle=True)
    snapshots = {t for r in labels.extra["replans"] for t in r["snapshot_ticks"]}
    assert set(frames["frame_ticks"].tolist()) == snapshots
