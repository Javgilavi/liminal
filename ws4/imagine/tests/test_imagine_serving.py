"""The policy server path on CPU: real ``flux.remote`` HTTP server and client, ``ImagineService``, the runner's
recorder and the replay script, with the fake FLUX policy behind a backend shaped like ``LeRobotFlux3Backend``."""

import json
import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from flux.backends import FakeBackend, PolicyRequest, PolicyResponse  # noqa: E402
from flux.contract import SO101_CONTRACT  # noqa: E402
from flux.remote import RemoteBackend, _unpack, encode_request, make_server  # noqa: E402

from ws4.imagine.attempt import AttemptImagination, load_attempt, render_attempt  # noqa: E402
from ws4.imagine.fakes import FakeConfig, FakeFluxPolicy  # noqa: E402
from ws4.imagine.serving import ImagineService, imagination_from_payload  # noqa: E402

N = SO101_CONTRACT.n_obs_steps


class FakeImaginingBackend:
    """``LeRobotFlux3Backend``'s structure (model lock, imagine branch, fetch) around the fake policy."""

    def __init__(self):
        self.contract = SO101_CONTRACT
        self.policy = FakeFluxPolicy(FakeConfig(canvas_hw=(64, 128)))
        self._lock = threading.Lock()
        self.service = ImagineService(self.policy, self._lock)

    def predict(self, request):
        batch = {"observation.state": torch.zeros(1, N, 6), "task": [request.task]}
        for key in self.contract.camera_keys:
            batch[key] = torch.from_numpy(request.images[key]).permute(0, 3, 1, 2)[None].float() / 255
        imagination = None
        with self._lock:
            if request.imagine:
                actions, meta, arrays = self.service.predict(batch, request.imagine)
                imagination = {"meta": meta, "arrays": arrays}
            else:
                actions = self.policy.predict_action_chunk(batch)
        commands = actions[0].double().numpy() + request.commands[-1]
        return PolicyResponse(commands, 0.0, {"backend": "fake-imagining"}, imagination)

    def fetch_imagination(self, imagination_id, wait_s=30.0):
        return self.service.fetch(imagination_id, wait_s)


def request(task="pick the pen", seed=0, imagine=""):
    rng = np.random.default_rng(seed)
    images = {k: rng.integers(0, 255, (2, 96, 128, 3), dtype=np.uint8) for k in SO101_CONTRACT.camera_keys}
    return PolicyRequest(task, np.zeros((N, 6)), np.ones((N, 6)), images, SO101_CONTRACT.snapshots, imagine=imagine)


@pytest.fixture
def server():
    backends = {"so101": FakeImaginingBackend(), "plain": FakeBackend(SO101_CONTRACT, velocity=0.1)}
    srv = make_server(backends, port=0, token="t0k")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", backends
    srv.shutdown()
    backends["so101"].service.close()


def test_plain_requests_are_unchanged_on_the_wire():
    _, meta = _unpack(encode_request(request()))
    assert "imagine" not in meta
    _, meta = _unpack(encode_request(request(imagine="lazy")))
    assert meta["imagine"] == "lazy"


@pytest.mark.parametrize("mode", ["sync", "async", "lazy"])
def test_modes_return_identical_actions_and_frames(server, mode):
    url, _ = server
    plain = RemoteBackend(url, "so101", token="t0k").predict(request())
    resp = RemoteBackend(url, "so101", token="t0k", imagine=mode).predict(request())
    assert plain.imagination is None
    assert np.array_equal(plain.commands, resp.commands)
    assert resp.info["imagination_id"] == resp.imagination["meta"]["id"]
    if mode == "sync":
        payload = resp.imagination
    else:
        assert "frames" not in resp.imagination["arrays"]
        payload = RemoteBackend(url, "so101", token="t0k").fetch_imagination(resp.info["imagination_id"])
    rec = imagination_from_payload(payload["meta"], payload["arrays"])
    assert rec.frames.shape == (41, 64, 128, 3) and rec.cond_frames.shape == (2, 64, 128, 3)
    assert rec.frame_offsets == tuple(range(2, 43)) and rec.task == "pick the pen"
    assert payload["meta"]["decode_s"] >= 0


def test_fetch_errors(server):
    url, _ = server
    remote = RemoteBackend(url, "so101", token="t0k")
    with pytest.raises(RuntimeError, match="404"):
        remote.fetch_imagination("nope")
    with pytest.raises(RuntimeError, match="404"):
        RemoteBackend(url, "plain", token="t0k").fetch_imagination("x")


def test_attempt_recorder_fetches_saves_and_renders(server, tmp_path):
    url, _ = server
    remote = RemoteBackend(url, "so101", token="t0k", imagine="lazy")
    irec = AttemptImagination(tmp_path)
    rng = np.random.default_rng(1)
    for tick in range(70):
        replanned = tick in (3, 35)  # the arm waited 3 ticks for the first chunk
        resp = remote.predict(request(seed=tick)) if replanned else None
        raw = {"scene": rng.integers(0, 255, (48, 64, 3), dtype=np.uint8),
               "wrist": rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)}  # fmt: skip
        irec.tick(tick, raw, replanned, resp)
    summary = irec.finish(remote, render=True, task="pick the pen")
    assert [c["step"] for c in summary["chunks"]] == [3, 35] and not any("error" in c for c in summary["chunks"])
    assert summary["ticks"] == 70 and (tmp_path / "imagination" / "imagined_vs_actual.mp4").exists()
    records, actual = load_attempt(tmp_path)
    assert [r.step for r in records] == [3, 35] and len(actual) == 70 and actual[5].image.shape == (256, 512, 3)
    assert render_attempt(tmp_path, gif=True).exists()
    from ws4.imagine.attempt import offset_errors

    errors = offset_errors(tmp_path)
    assert set(errors) == {"scene", "wrist", "canvas"} and min(errors["canvas"]) == 2


def test_replay_server_end_to_end(server, tmp_path, monkeypatch):
    from ws4.imagine import replay_server

    url, _ = server
    n = 120
    rng = np.random.default_rng(2)

    class FakeDataset:
        def __len__(self):
            return n

        def __getitem__(self, i):
            return {k: torch.rand(3, 48, 64, generator=torch.Generator().manual_seed(i))
                    for k in ("observation.images.scene", "observation.images.wrist")}  # fmt: skip

    states = rng.normal(0, 10, (n, 6))
    monkeypatch.setattr(replay_server, "load_episode", lambda *a, **k: (states, states + 1, FakeDataset()))
    token = tmp_path / "token"
    token.write_text("t0k")
    report = replay_server.main(["--url", url, "--token-file", str(token), "--repo-id", "fake",
                                 "--task", "pick the pen", "--chunks", "2", "--out", str(tmp_path / "out")])
    assert report["all_identical"] and len(report["chunks"]) == 2
    assert [c["tick"] for c in report["chunks"]] == [8, 40]
    assert (tmp_path / "out" / "imagination" / "imagined_vs_actual.mp4").exists()
    assert json.loads((tmp_path / "out" / "report.json").read_text())["median_imagine_round_trip_s"] > 0


def test_recorder_tick_never_raises_or_blocks(tmp_path, monkeypatch, capsys):
    import time as _time

    from ws4.imagine import attempt

    slow = lambda *a, **k: _time.sleep(0.05) or np.zeros((256, 512, 3), np.uint8)  # noqa: E731
    monkeypatch.setattr(attempt, "make_canvas", slow)
    irec = AttemptImagination(tmp_path, max_queue=2)
    frame = {"scene": np.zeros((48, 64, 3), np.uint8), "wrist": np.zeros((48, 64, 3), np.uint8)}
    t0 = _time.perf_counter()
    for tick in range(50):
        irec.tick(tick, frame)
    assert _time.perf_counter() - t0 < 0.5 and irec.dropped > 0  # dropped frames, never waited
    irec.tick(50, None)  # a broken observation disables recording with one line
    irec.tick(51, None)
    assert irec.disabled and capsys.readouterr().out.count("disabled") == 1


def test_finish_budget_errors_and_later_fetch(server, tmp_path):
    from ws4.imagine.attempt import fetch_pending, finish_with_timeout

    url, _ = server
    remote = RemoteBackend(url, "so101", token="t0k", imagine="lazy")
    irec = AttemptImagination(tmp_path)
    frame = {"scene": np.zeros((48, 64, 3), np.uint8), "wrist": np.zeros((48, 64, 3), np.uint8)}
    for tick in range(70):
        irec.tick(tick, frame, tick in (1, 34), remote.predict(request(seed=tick)) if tick in (1, 34) else None)

    class Down:
        def fetch_imagination(self, *a):
            raise ConnectionRefusedError("server down")

    summary = finish_with_timeout(irec, Down(), timeout_s=10)
    assert summary["pending"] == 2 and "error" in summary["chunks"][0] and summary["chunks"][1]["pending"]
    later = fetch_pending(tmp_path, remote)  # the server still holds the latents
    assert later["pending"] == 0 and len(load_attempt(tmp_path)[0]) == 2


def test_finish_with_timeout_returns_on_a_hanging_server(tmp_path):
    import time as _time

    from ws4.imagine.attempt import finish_with_timeout

    class Hang:
        def fetch_imagination(self, *a):
            _time.sleep(30)

    irec = AttemptImagination(tmp_path)
    irec.chunks.append({"step": 0, "t": 0.0, "imagination": {"meta": {"id": "x"}, "arrays": {}},
                        "round_trip_s": 0.1, "server_latency_s": 0.1})  # fmt: skip
    t0 = _time.perf_counter()
    assert finish_with_timeout(irec, Hang(), timeout_s=1.0) is None
    assert _time.perf_counter() - t0 < 3
