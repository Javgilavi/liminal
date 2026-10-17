import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ws4.imagine.fakes import FakeConfig, FakeFluxPolicy, FakeVideoVAE, fake_batch, sampling  # noqa: E402
from ws4.imagine.predict_with_video import (  # noqa: E402
    Imaginer,
    LatentDecoder,
    alternatives_batch,
    capture_solver_outputs,
    load_imagination,
)


def test_imagine_returns_stock_action_and_frames():
    policy = FakeFluxPolicy()
    stock = policy.predict_action_chunk(fake_batch())
    seen = []
    (rec,) = Imaginer(policy, on_imagination=seen.append).imagine(fake_batch())
    assert torch.equal(rec.action, stock[0])
    assert rec.frames.shape == (41, 64, 128, 3) and rec.frames.dtype == np.uint8
    assert rec.cond_frames.shape == (2, 64, 128, 3)
    assert rec.latents.shape == (96, 11, 2, 4)
    assert seen == [rec]
    assert not {"_sample", "predict_action_chunk", "select_action"} & set(policy.__dict__)
    assert rec.views(rec.final_frame())["scene"].shape == (64, 64, 3)


def test_batch_items_get_one_record_each():
    recs = Imaginer(FakeFluxPolicy()).imagine(fake_batch(batch_size=3))
    assert [r.index for r in recs] == [0, 1, 2]
    assert all(r.frames is not None for r in recs)


def test_installed_policy_imagines_inside_select_action_with_async_decode():
    policy = FakeFluxPolicy()
    frames_ready = []
    imaginer = Imaginer(policy, decode="async", on_frames=frames_ready.append).install()
    policy.reset()
    for _ in range(65):
        policy.select_action(fake_batch())
    assert [r.step for r in imaginer.history] == [0, 32, 64]
    for rec in imaginer.history:
        rec.wait(10)
        assert rec.frames.shape[0] == 41
    imaginer.close()
    assert len(frames_ready) == 3
    assert "select_action" not in policy.__dict__


def test_replan_keeps_steps_monotonic():
    policy = FakeFluxPolicy()
    imaginer = Imaginer(policy).install()
    for _ in range(10):
        policy.select_action(fake_batch())
    imaginer.replan()  # e.g. the surprise monitor fired
    policy.reset()  # an orchestrator flushing the queue the LeRobot way must not restart step counting
    policy.select_action(fake_batch())
    assert [r.step for r in imaginer.history] == [0, 10]
    imaginer.new_episode()
    policy.reset()
    policy.select_action(fake_batch())
    assert [r.step for r in imaginer.history] == [0]
    imaginer.uninstall()


def test_best_of_n_picks_the_highest_score_and_its_frames():
    policy = FakeFluxPolicy()
    calls = []

    def scorer(clips, *, current, offsets, task, spec):
        calls.append((len(clips), current.shape, len(offsets), task))
        return [0.1, 0.9, 0.5]

    (rec,) = Imaginer(policy, n_candidates=3, scorer=scorer).imagine(fake_batch(task="stack"))
    assert calls == [(3, (64, 128, 3), 41, "stack")]
    assert [c.seed for c in rec.candidates] == [42, 43, 44]
    assert rec.chosen == 1 and rec.seed == 43 and rec.score == pytest.approx(0.9)
    assert rec.frames is rec.candidate_frames[1]
    ref = FakeFluxPolicy(FakeConfig(inference_seed=43)).predict_action_chunk(fake_batch(task="stack"))
    assert torch.equal(rec.action, ref[0])


def test_latent_scorer_needs_no_decoding():
    class Latent:
        space = "latent"

        def __call__(self, latents, **kw):
            return [float(z.mean()) for z in latents]

    policy = FakeFluxPolicy()
    (rec,) = Imaginer(policy, n_candidates=2, scorer=Latent(), decode="none").imagine(fake_batch())
    assert policy.frozen.video_vae.decode_calls == 0
    assert rec.frames is None and len(rec.candidates) == 2


def test_frame_conditioning_decodes_after_the_condition_latent():
    cfg = FakeConfig(conditioning="frame", chunk_size=32, fps=15.0)
    (rec,) = Imaginer(FakeFluxPolicy(cfg)).imagine(fake_batch(cfg))
    assert rec.frame_offsets == tuple(range(1, 33))
    assert rec.frames.shape[0] == 32 and rec.cond_frames.shape[0] == 1


def test_capture_is_thread_local():
    results = {}

    def worker(name, value):
        with capture_solver_outputs(sampling) as outs:
            sampling.euler({"x": torch.full((1,), value)}, lambda s, t: {"x": torch.zeros(1)}, n_steps=1, alpha=1.0)
        results[name] = [float(o["x"]) for o in outs]

    threads = [threading.Thread(target=worker, args=(n, v)) for n, v in (("a", 1.0), ("b", 2.0))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == {"a": [1.0], "b": [2.0]}


def test_save_and_load_round_trip(tmp_path):
    (rec,) = Imaginer(FakeFluxPolicy()).imagine(fake_batch())
    back = load_imagination(rec.save(tmp_path / "chunk.npz"))
    assert back.spec == rec.spec and back.task == rec.task
    assert np.array_equal(back.frames, rec.frames)
    assert np.allclose(back.action, rec.action.numpy())


def test_alternative_instructions_from_one_observation():
    batch = fake_batch()
    recs = Imaginer(FakeFluxPolicy()).imagine(alternatives_batch(batch, ["push left", "push right", "lift"]))
    assert [r.task for r in recs] == ["push left", "push right", "lift"]
    with pytest.raises(ValueError):
        alternatives_batch(fake_batch(batch_size=2), ["a", "b"])


def test_best_of_n_requires_a_scorer():
    with pytest.raises(ValueError):
        Imaginer(FakeFluxPolicy(), n_candidates=2)


class DeviceSpyVAE(FakeVideoVAE):
    """Weights on another device (``meta`` stands in for CUDA on a CPU-only laptop); records where latents land."""

    def __init__(self):
        super().__init__()
        self.module = torch.nn.Linear(1, 1, device="meta")
        self.seen = []

    def decode(self, latents):
        self.seen.append(latents.device.type)
        return super().decode(torch.zeros(latents.shape, dtype=latents.dtype))


def test_decoder_moves_cpu_latents_to_the_vae_device():
    # Regression: captured latents are on CPU; with device=None they must follow the VAE, as LeRobot's decode
    # calls torch.cuda.mem_get_info(z.device) and rejects CPU tensors (seen on a RunPod GPU).
    vae = DeviceSpyVAE()
    LatentDecoder(vae).decode_latents(torch.zeros(1, 96, 2, 2, 4))
    LatentDecoder(vae, device="cpu").decode_latents(torch.zeros(1, 96, 2, 2, 4))
    assert vae.seen == ["meta", "cpu"]
    policy = FakeFluxPolicy()
    policy.frozen.video_vae = DeviceSpyVAE()
    (rec,) = Imaginer(policy).imagine(fake_batch())  # the default decoder, same GPU as the policy
    assert set(policy.frozen.video_vae.seen) == {"meta"} and rec.frames.shape[0] == 41
