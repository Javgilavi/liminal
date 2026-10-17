"""ACT backend on a tiny random ACT policy (CPU only; no GPU, network, weights download or serial port)."""

import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot.policies.act.modeling_act")

from act.backend import ActBackend, contract_from_act_config  # noqa: E402
from flux.backends import PolicyRequest  # noqa: E402
from flux.contract import IMG_HEAD, IMG_RIGHT_WRIST, OBS_STATE, SO101_SCENE, SO101_WRIST  # noqa: E402
from flux.dual_arm_runner import DualArmRunner, per_arm_target  # noqa: E402
from flux.remote import RemoteBackend, make_server  # noqa: E402
from flux.safety import SafetyConfig, SafetyGuard  # noqa: E402

HW = (48, 64)
ACTION_MEAN, ACTION_STD = 40.0, 10.0


def tiny_act(chunk: int = 10, n_action_steps: int = 10):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    torch.manual_seed(0)
    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, *HW))
    config = ACTConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)), SO101_SCENE: image,
                        SO101_WRIST: image},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        chunk_size=chunk, n_action_steps=n_action_steps, dim_model=32, n_heads=2, dim_feedforward=64,
        n_encoder_layers=1, n_decoder_layers=1, n_vae_encoder_layers=1, latent_dim=4,
        pretrained_backbone_weights=None, device="cpu", push_to_hub=False,
    )  # fmt: skip
    policy = ACTPolicy(config)
    img_stats = {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.25)}
    stats = {
        OBS_STATE: {"mean": torch.zeros(6), "std": torch.full((6,), 30.0)},
        "action": {"mean": torch.full((6,), ACTION_MEAN), "std": torch.full((6,), ACTION_STD)},
        SO101_SCENE: img_stats,
        SO101_WRIST: img_stats,
    }
    pre, post = make_act_pre_post_processors(config, dataset_stats=stats)
    return policy, pre, post


@pytest.fixture(scope="module")
def backend():
    return ActBackend(*tiny_act(), name="tiny-act")


def request(contract, state=None, value=100, ticks=None):
    n = contract.n_obs_steps if ticks is None else ticks
    state = np.zeros(6) if state is None else np.asarray(state, float)
    shape = (len(contract.snapshots), *contract.image_hw, 3)
    images = {k: np.full(shape, value, np.uint8) for k in contract.camera_keys}
    return PolicyRequest("put the red pen in its slot in the cardboard box", np.tile(state, (n, 1)),
                         np.tile(state, (n, 1)), images, contract.snapshots)  # fmt: skip


def test_contract_is_one_tick_full_resolution(backend):
    c = backend.contract
    assert (c.n_obs_steps, c.history_snapshots, c.snapshots) == (1, 1, (0,))
    assert c.camera_keys == (SO101_SCENE, SO101_WRIST) and c.image_hw == HW
    assert (c.chunk_size, c.n_action_steps, c.action_dim) == (10, 10, 6)
    assert c.absolute_dims == tuple(range(6)) and c.output_space == "joints"
    assert contract_from_act_config(backend.policy.config, "x", n_action_steps=4).n_action_steps == 4


def test_predict_returns_absolute_unnormalized_chunk(backend):
    req = request(backend.contract, state=[1, 2, 3, 4, 5, 50])
    out = backend.predict(req)
    assert out.commands.shape == (10, 6) and out.commands.dtype == np.float64 and np.isfinite(out.commands).all()
    # The same chunk as calling the policy by hand, mapped back to dataset units by the saved post-processor.
    with torch.inference_mode():
        normalized = backend.policy.predict_action_chunk(backend.pre(backend.batch(req)))
    assert np.allclose(out.commands, normalized[0].numpy() * ACTION_STD + ACTION_MEAN, atol=1e-4)
    assert np.allclose(backend.predict(req).commands, out.commands)  # eval mode: no sampling, no queue state
    assert out.info["backend"] == "lerobot_act"


def test_predict_uses_the_current_frame_and_state(backend):
    a = backend.predict(request(backend.contract, value=0)).commands
    b = backend.predict(request(backend.contract, value=255)).commands
    c = backend.predict(request(backend.contract, state=[30, -20, 10, 0, 5, 80], value=0)).commands
    assert not np.allclose(a, b) and not np.allclose(a, c)


def test_rejects_windows_it_was_not_built_for(backend):
    with pytest.raises(ValueError, match="states/commands"):
        backend.predict(request(backend.contract, ticks=8))
    bad = request(backend.contract)
    bad.images[SO101_WRIST] = bad.images[SO101_WRIST][..., :1]
    with pytest.raises(ValueError, match="uint8"):
        backend.predict(bad)


class ReplayRobot:
    """12-D station robot whose state follows the guarded command; cameras are constant 480x640 frames."""

    def __init__(self):
        self.state = np.zeros(12)
        self.state[11] = 100.0  # haller home: gripper open
        self.sent = []

    def get_observation(self):
        obs = {OBS_STATE: self.state.copy()}
        obs.update({k: np.full((480, 640, 3), 90, np.uint8) for k in (IMG_HEAD, IMG_RIGHT_WRIST)})
        return obs

    def send_action(self, action):
        self.sent.append(np.array(action))
        self.state = np.array(action, float)


def test_runner_executes_act_chunks_through_the_guard():
    backend = ActBackend(*tiny_act(chunk=10, n_action_steps=4), name="tiny-act")
    robot = ReplayRobot()
    step = np.full(12, 1.0)
    guard = SafetyGuard(SafetyConfig(lower=np.full(12, -150.0), upper=np.full(12, 150.0), max_step=step))
    runner = DualArmRunner(robot, [per_arm_target("right", "act")], {"act": backend}, guard)
    runner.start("right", "put the red pen in its slot in the cardboard box")
    reports = [runner.step() for _ in range(9)]
    # Re-plans every n_action_steps ticks; the runner resized the 480x640 frames to the checkpoint's 48x64.
    assert [r.status["right"] for r in reports].count("replanned") == 3
    assert runner._targets["right"].n_chunks == 3  # noqa: SLF001
    sent = np.array(robot.sent)
    prev = np.vstack([np.r_[np.zeros(11), 100.0], sent[:-1]])
    assert (np.abs(sent - prev) <= 1.0 + 1e-9).all()  # never more than the guard's 1 deg per tick
    assert np.allclose(sent[:, :6], 0.0)  # the left arm is not owned by the target and holds
    assert any(r.guard_intervened for r in reports)  # random ACT targets are far away: the guard ramps them


def test_served_over_flux_remote_protocol(backend):
    server = make_server({"act": backend}, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        remote = RemoteBackend(f"http://127.0.0.1:{server.server_address[1]}", "act", codec="raw")
        assert remote.contract == backend.contract
        req = request(backend.contract, state=[1, 2, 3, 4, 5, 50])
        assert np.allclose(remote.predict(req).commands, backend.predict(req).commands)
    finally:
        server.shutdown()


def test_loads_a_saved_checkpoint_with_newer_lerobot_fields(tmp_path, backend):
    import json

    backend.policy.save_pretrained(tmp_path)
    backend.pre.save_pretrained(tmp_path)
    backend.post.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    config.update(dtype=None, pretrained_revision=None, pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1")
    (tmp_path / "config.json").write_text(json.dumps(config))  # as LeRobot 146d847 on the pod writes it
    loaded = ActBackend.from_pretrained(str(tmp_path), device="cpu", n_action_steps=5)
    assert loaded.policy.config.pretrained_backbone_weights is None  # no torchvision download at load
    assert loaded.contract.n_action_steps == 5
    req = request(backend.contract, state=[1, 2, 3, 4, 5, 50])
    assert np.allclose(loaded.predict(req).commands, backend.predict(req).commands, atol=1e-5)
    config["future_field"] = "bfloat16"  # a field this LeRobot does not know, with a value
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unknown to this LeRobot"):
        ActBackend.from_pretrained(str(tmp_path), device="cpu")


def test_single_arm_runner_drives_act_over_the_remote_protocol(tmp_path, monkeypatch):
    """run_single_arm.py unchanged (--backend remote, fake arm) against an ACT server on 127.0.0.1."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "single_arm"))
    import run_single_arm as rsa

    act = ActBackend(*tiny_act(chunk=10, n_action_steps=4), name="tiny-act")
    server = make_server({"act": act}, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(rsa, "RUNS", tmp_path)
    monkeypatch.setattr(rsa, "STOP_FILE", tmp_path / "STOP")
    monkeypatch.setattr(rsa, "KEY_FILE", tmp_path / "KEY")
    try:
        rsa.main(["--robot", "fake", "--backend", "remote", "--server", f"http://127.0.0.1:{server.server_address[1]}",
                  "--package", "act", "--seconds", "1.5", "--yes", "--no-video", "--no-log", "--no-keepout",
                  "--max-step", "0.5", "--task", "put the red pen in its slot in the cardboard box"])  # fmt: skip
    finally:
        server.shutdown()
    meta = json.loads(next(tmp_path.glob("*/attempt01/meta.json")).read_text())
    assert meta["end_reason"] == "timeout" and meta["chunks"] >= 2  # re-plans every 4 executed commands
    trace = np.load(next(tmp_path.glob("*/attempt01/trace.npz")))
    assert np.abs(np.diff(trace["command"][:, :5], axis=0)).max() <= 0.5 + 1e-9  # guard limit on every tick
    assert trace["guard"].any()


def constant_act(pose, chunk: int = 10):
    """Tiny ACT whose every command is ``pose``: the post-processor's std is ~0, so outputs collapse to its mean."""
    policy, pre, post = tiny_act(chunk=chunk, n_action_steps=chunk)
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    img_stats = {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.25)}
    stats = {
        OBS_STATE: {"mean": torch.zeros(6), "std": torch.full((6,), 30.0)},
        "action": {"mean": torch.tensor(pose, dtype=torch.float32), "std": torch.full((6,), 1e-6)},
        SO101_SCENE: img_stats,
        SO101_WRIST: img_stats,
    }
    pre, post = make_act_pre_post_processors(policy.config, dataset_stats=stats)
    return ActBackend(policy, pre, post, name="constant-act")


def run_haller_fake(rsa, tmp_path, monkeypatch, policy_args, seconds="4"):
    """run_single_arm.main on the fake arm from haller home, Haller keep-out with the final-split demo envelope."""
    import json
    from pathlib import Path

    class HomeFake(rsa.FakeSO101):
        def __init__(self, start=None):
            super().__init__(start=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 100.0]))

    keepout = Path(rsa.HERE) / "calibration" / "haller_right_workspace_final.json"
    monkeypatch.setattr(rsa, "FakeSO101", HomeFake)
    monkeypatch.setattr(rsa, "RUNS", tmp_path)
    monkeypatch.setattr(rsa, "STOP_FILE", tmp_path / "STOP")
    monkeypatch.setattr(rsa, "KEY_FILE", tmp_path / "KEY")
    rsa.main(["--robot", "fake", *policy_args, "--keepout-config", str(keepout), "--start-pose", "0", "0", "0", "0",
              "0", "100", "--seconds", seconds, "--yes", "--no-video", "--no-log", "--max-step", "5",
              "--task", "put the red pen in its slot in the cardboard box"])  # fmt: skip
    meta = json.loads(next(tmp_path.glob("*/attempt01/meta.json")).read_text())
    return meta, np.load(next(tmp_path.glob("*/attempt01/trace.npz")))


def test_a_bad_act_command_is_refused_by_the_keepout(tmp_path, monkeypatch):
    """ACT asks for pan 120 deg (demos reach 57.5): the demo envelope refuses it (the TCP envelope first, at about
    10 deg from haller home), the arm holds, and 45 refused ticks in a row end the attempt. ACT's full 100-step
    chunk matters here: the runner's "waiting" ticks between chunks reset that count."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "single_arm"))
    import run_single_arm as rsa

    server = make_server({"act": constant_act([120.0, 0.0, 0.0, 0.0, 0.0, 100.0], chunk=100)}, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        meta, trace = run_haller_fake(rsa, tmp_path, monkeypatch, ["--backend", "remote", "--server", url,
                                                                   "--package", "act"])  # fmt: skip
    finally:
        server.shutdown()
    assert meta["end_reason"].startswith("stop (keep-out: ") and "refused for 45 ticks" in meta["end_reason"]
    refusals = meta["keepout_refusals"]
    assert refusals["joint_env"] + refusals["tcp_env"] >= 45
    assert trace["command"][:, 0].max() < 57.6  # never past the demonstrated pan range
    assert np.abs(np.diff(trace["command"][:, :5], axis=0)).max() <= 5.0 + 1e-9


def test_backend_act_in_process_is_refused_the_same_way_and_logs_its_checkpoint(tmp_path, monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "single_arm"))
    import run_single_arm as rsa

    bad = constant_act([120.0, 0.0, 0.0, 0.0, 0.0, 100.0], chunk=100)
    ckpt = tmp_path / "ckpt"
    for part in (bad.policy, bad.pre, bad.post):
        part.save_pretrained(ckpt)
    runs = tmp_path / "runs"
    runs.mkdir()
    meta, trace = run_haller_fake(rsa, runs, monkeypatch, ["--backend", "act", "--act-checkpoint", str(ckpt),
                                                           "--act-device", "cpu"])  # fmt: skip
    assert meta["end_reason"].startswith("stop (keep-out: ") and "refused for 45 ticks" in meta["end_reason"]
    assert trace["command"][:, 0].max() < 57.6
    assert meta["checkpoint"] == f"{ckpt} (LeRobot ACT)"  # the trial log names the ACT checkpoint, not FLUX's
