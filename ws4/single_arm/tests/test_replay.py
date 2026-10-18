"""--backend replay (replay.py): one recorded take as a scripted policy through the runner's full safety stack, and
--prime-ticks (the first ticks from the take, then FLUX). Fake arm and a synthetic LeRobot parquet; the real box
take is checked too when the dataset is in the local cache."""

import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import replay  # noqa: E402
import run_single_arm as rsa  # noqa: E402
from flux.backends import FakeBackend, PolicyRequest  # noqa: E402
from flux.contract import SO101_CONTRACT  # noqa: E402
from flux.remote import make_server  # noqa: E402

BOX_HOME = np.array([0, 0.8, 2.0, 0.4, 0, 99.7])
# a short "box" take for the robot-right arm (dataset columns left_*): out from home, roll 50 deg, then hold
TAKE = np.array([[0, 0, 0, 0, 0, 100.0]] + [[i * 0.6, -i * 0.4, i * 0.9, 0.0, i * 2.5, 100.0] for i in range(1, 21)]
                + [[12.0, -8.0, 18.0, 0.0, 50.0, 100.0]] * 10)


def write_dataset(root: Path, episodes: dict[int, np.ndarray], other: np.ndarray | None = None) -> Path:
    """LeRobot v3 layout, data columns only: 12-D actions [left_* | right_*], the take in the left_* columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {"episode_index": [], "frame_index": [], "action": []}
    for ep, take in episodes.items():
        pen = np.tile(other if other is not None else [3, -83.6, 81.9, 82.8, -10.2, 0.9], (len(take), 1))
        for i, a in enumerate(np.hstack([take, pen])):
            rows["episode_index"].append(ep)
            rows["frame_index"].append(i)
            rows["action"].append([float(v) for v in a])
    (root / "data" / "chunk-000").mkdir(parents=True)
    table = pa.table({"episode_index": pa.array(rows["episode_index"], pa.int64()),
                      "frame_index": pa.array(rows["frame_index"], pa.int64()),
                      "action": pa.array(rows["action"], pa.list_(pa.float32()))})
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    return root


def test_time_warp_follows_the_take_within_the_step_limit():
    seq = replay.time_warp(TAKE, BOX_HOME, max_step=1.0, max_gripper_step=5.0)
    steps = np.abs(np.diff(np.vstack([BOX_HOME, seq]), axis=0))
    assert steps[:, :5].max() <= 1.0 + 1e-9 and steps[:, 5].max() <= 5.0 + 1e-9
    assert np.allclose(seq[-1], TAKE[-1])
    for a in TAKE:  # every recorded action is on the warped path
        assert np.abs(seq - a).max(axis=1).min() < 1e-9
    assert len(seq) > len(TAKE)  # slower where the take moved faster than 1 deg per frame


def test_load_take_reads_one_episode_of_one_arm(tmp_path):
    root = write_dataset(tmp_path / "box", {3: TAKE, 4: TAKE[::-1]})
    actions, info = replay.load_take(root, 3, "left")
    assert np.allclose(actions, TAKE, atol=1e-4) and info["frames"] == len(TAKE) and info["episode"] == 3
    pen, _ = replay.load_take(root, 3, "right")
    assert np.allclose(pen[0], [3, -83.6, 81.9, 82.8, -10.2, 0.9], atol=1e-4)
    with pytest.raises(SystemExit, match="episode 9 is not in"):
        replay.load_take(root, 9, "left")


def request(anchor, target="policy"):
    c = SO101_CONTRACT
    images = {k: np.zeros((c.history_snapshots, *c.image_hw, 3), np.uint8) for k in c.camera_keys}
    hist = np.repeat(np.asarray(anchor, float)[None], c.n_obs_steps, 0)
    return PolicyRequest("open the box by pulling its handle", hist, hist, images, c.snapshots, target=target)


def test_replay_backend_hands_out_the_take_then_holds_and_says_done():
    be = replay.ReplayBackend(TAKE, {"episode": 7, "root": "x", "columns": "left", "frames": len(TAKE)})
    assert be.label == "scripted replay of recorded demo 7"
    warm = be.predict(request(BOX_HOME, target="warmup"))  # a warm-up never advances the take
    assert be.seq is None and np.allclose(warm.commands, BOX_HOME)
    seq, out = replay.time_warp(TAKE, BOX_HOME), []
    while not be.done:
        r = be.predict(request(BOX_HOME))
        assert r.info["backend"] == "replay" and r.info["label"] == be.label
        out.extend(r.commands[: SO101_CONTRACT.n_action_steps])
    assert np.allclose(np.asarray(out)[: len(seq)], seq)  # the executed parts of the chunks are the take in order
    assert np.allclose(np.asarray(out)[len(seq):], TAKE[-1])  # then it holds where the take ends
    assert be.finished() and "holding at its end" in be.progress()[2]
    be.begin("right:replay_open_box", "right", None)
    assert not be.done and be.seq is None


def _run(tmp_path, monkeypatch, *extra):
    monkeypatch.setattr(rsa, "RUNS", tmp_path)
    monkeypatch.setattr(rsa, "STOP_FILE", tmp_path / "STOP")
    monkeypatch.setattr(rsa, "KEY_FILE", tmp_path / "KEY")
    monkeypatch.setenv("WS4_TRIAL_LOG", str(tmp_path / "trials.jsonl"))
    rsa.main(["--robot", "fake", "--yes", "--no-video", "--task-side", "right", "--max-step", "1.0", *extra])
    return [json.loads(m.read_text()) for m in sorted(tmp_path.glob("*/attempt*/meta.json"))]


def test_replay_refuses_settings_that_would_break_it(tmp_path, monkeypatch):
    root = write_dataset(tmp_path / "box", {5: TAKE})
    with pytest.raises(SystemExit):
        _run(tmp_path / "runs", monkeypatch, "--backend", "replay", "--replay-root", str(root), "--replay-episode",
             "5", "--scale", "1", "-1", "1", "1", "-1", "1")  # fmt: skip
    with pytest.raises(SystemExit, match="needs --backend remote"):
        _run(tmp_path / "runs", monkeypatch, "--backend", "fake", "--prime-ticks", "32", "--replay-root", str(root),
             "--replay-episode", "5")  # fmt: skip


def test_prefetch_is_forced_off_for_a_replay(tmp_path, monkeypatch, capsys):
    root = write_dataset(tmp_path / "box", {5: TAKE})
    monkeypatch.setenv("WS4_PREFETCH_TICKS", "16")  # set for the FLUX rows; it would skip recorded steps here
    (meta,) = _run(tmp_path / "runs", monkeypatch, "--backend", "replay", "--replay-root", str(root),
                   "--replay-episode", "5", "--seconds", "60", "--no-keepout")  # fmt: skip
    assert meta["args"]["prefetch_ticks"] == 0 and meta["end_reason"].startswith("replay finished")
    assert "chunk prefetch off for the replay (was 16)" in capsys.readouterr().out


def test_prime_ticks_replays_the_start_then_hands_over_to_flux(tmp_path, monkeypatch):
    root = write_dataset(tmp_path / "box", {5: TAKE})
    flux = FakeBackend(SO101_CONTRACT, velocity=0.0)
    server = make_server({"skill_open_box": flux}, port=0, token="secret")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        (meta,) = _run(tmp_path / "runs", monkeypatch, "--backend", "remote", "--package", "skill_open_box",
                       "--server", f"http://127.0.0.1:{server.server_address[1]}", "--token-cmd", "echo secret",
                       "--no-log", "--no-keepout", "--seconds", "4", "--prime-ticks", "32", "--replay-root",
                       str(root), "--replay-episode", "5", "--fake-delay", "0.0", "--start-pose",
                       *map(str, BOX_HOME))  # fmt: skip
    finally:
        server.shutdown()
        server.server_close()
    assert meta["replay"]["handed_over_to_flux"] and meta["replay"]["ticks_sent"] == 32
    assert meta["replay"]["label"] == "scripted replay of recorded demo 5 for the first 32 ticks, then FLUX"
    assert meta["end_reason"] == "timeout" and len(flux.requests) >= 2  # its warm-up, then the chunks after tick 32
    seq = replay.time_warp(TAKE, np.array(meta["start_q"]))  # the fake arm skips the ramp: from its rest pose
    trace = np.load(next((tmp_path / "runs").glob("*/attempt01/trace.npz")))
    executed = trace["command"][trace["status"] != "waiting"]  # without the holds while a chunk is pending
    assert np.allclose(executed[:32], seq[:32], atol=1e-6)  # the first 32 commands were the take's
    assert not np.allclose(executed[32:40], seq[32:40], atol=1e-3)  # then FLUX's (the fake one holds)


BOX = Path(replay.DEFAULT_ROOT)


@pytest.mark.skipif(not (BOX / "data").exists(), reason="the box dataset is not in the local cache")
def test_default_box_take_replays_inside_the_box_workspace():
    """Take 59 time-warped from the chain's box home: no command the box skill's keep-out (with its envelope, the
    pen arm tucked) would refuse, inside one 120 s attempt, ending where take 59 ends."""
    from keepout import KeepOut

    actions, info = replay.load_take(BOX, replay.DEFAULT_EPISODE, "left")
    seq = replay.time_warp(actions, BOX_HOME)
    ko = KeepOut.load(HERE / "skills" / "box" / "workspace.json")
    ko.other_q = (3, -83.6, 81.9, 82.8, -10.2, 0.9)
    assert not any(ko.refuse(b, a)[0] for a, b in zip(np.vstack([BOX_HOME, seq])[:-1], seq, strict=True))
    assert len(seq) / 30 < 60 and info["frames"] == 409
