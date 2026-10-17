"""FLUX 3 Action inference server. Runs on the GPU box (BF16 needs ~32 GB of GPU memory).

One stateful session: the saved LeRobot processors keep the 8-step observation/command history and
integrate the checkpoint's command deltas, so the client must send every control tick (30 Hz of
sim time) and call /reset between episodes.

    python server.py --port 8100                 # real model (see README for the install), >= ~32 GB GPU
    python server.py --port 8100 --offload       # real model on a small GPU (8 GB laptop): streams weights
    python server.py --port 8100 --fake          # no model: echoes the state back, to test the wiring

Protocol (JSON):
    POST /reset {"task": str}                                   -> {"ok": true}
    POST /act   {"task": str, "state": [6], "scene": b64 JPEG, "wrist": b64 JPEG}
                -> {"action": [6], "infer_ms": float | null}
State/action units: degrees for the five arm joints, 0-100 for the gripper (LeRobot SO-101 convention).
"""

from __future__ import annotations

import argparse
import base64
import io
import threading
import time

import numpy as np
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

REPO = "black-forest-labs/flux-3-action-so101"
REVISION = "c9e13b2aca6a0a472b3ea03fd90cff32d3e85849"  # pinned in the BFL SO-101 guide


class ResetReq(BaseModel):
    task: str = ""


class ActReq(BaseModel):
    task: str
    state: list[float]
    scene: str
    wrist: str


def _decode(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


class BlockStreamer:
    """Keeps `blocks` in CPU RAM and copies each one to the GPU only while its forward runs.

    Lets a model larger than GPU memory run at the cost of one host->device copy of its weights per
    forward pass. `pin=True` (page-locked RAM) roughly doubles copy speed but that RAM can't be swapped,
    so it's used for the transformer (every chunk) and not for the text encoder (once per prompt).
    """

    def __init__(self, blocks, device: str, pin: bool):
        self.device = device
        for block in blocks:
            tensors = [*block.parameters(), *block.buffers()]
            for t in tensors:
                t.data = t.data.contiguous().pin_memory() if pin else t.data
            block._stream_cpu = [(t, t.data) for t in tensors]
            block.register_forward_pre_hook(self._load)
            block.register_forward_hook(self._unload)

    def _load(self, block, args):
        for t, cpu in block._stream_cpu:
            t.data = cpu.to(self.device, non_blocking=True)

    def _unload(self, block, args, out):
        for t, cpu in block._stream_cpu:
            t.data = cpu


class FluxRunner:
    def __init__(self, repo: str, revision: str | None, device: str, offload: bool = False):
        print("Importing FLUX runtime...", flush=True)
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.flux3 import Flux3Policy

        self.torch = torch
        self.device = device
        self.offload = offload
        print(f"Loading {repo}@{revision} (offload={offload})...", flush=True)
        if offload:
            # Load everything on the CPU, then place it piecewise (see _place_offloaded).
            config = PreTrainedConfig.from_pretrained(repo, revision=revision)
            config.device = "cpu"
            self.policy = Flux3Policy.from_pretrained(repo, revision=revision, config=config)
            self.policy.config.device = device
            self._place_offloaded()
        else:
            self.policy = Flux3Policy.from_pretrained(repo, revision=revision)
            self.policy.to(device)
        self.policy.eval()
        print("Model weights loaded; restoring saved processors...", flush=True)
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=repo, pretrained_revision=revision)
        cfg = self.policy.config
        print(f"loaded {repo}@{revision}: cameras={cfg.camera_order} fps={cfg.fps} chunk={cfg.chunk_size} "
              f"execute={cfg.n_action_steps} history={cfg.n_obs_steps} offload={offload}")
        if device.startswith("cuda"):
            print(f"GPU memory in use after load: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    def _place_offloaded(self) -> None:
        """Video VAE on the GPU; transformer blocks streamed from pinned RAM, the rest of it on the GPU;
        text encoder left in RAM and streamed through the GPU only when a new prompt needs encoding.

        Never call policy.to() afterwards: Flux3Policy._apply would move all of it at once."""
        from lerobot.policies.flux3.f3.transformer import ModeBlock, SingleStreamBlock

        policy, dev = self.policy, self.device
        policy.frozen.video_vae.to(dev)
        blocks = []
        for name, child in policy.dit.named_children():
            if "blocks" in name:
                blocks += [m for m in child.modules() if isinstance(m, (ModeBlock, SingleStreamBlock))]
            else:
                child.to(dev)
        for _name, p in policy.dit.named_parameters(recurse=False):
            p.data = p.data.to(dev)
        BlockStreamer(blocks, dev, pin=True)
        print(f"streaming {len(blocks)} transformer blocks from pinned RAM")
        te = policy.frozen.text_encoder.model
        self._te_layers = te.model.language_model.layers
        BlockStreamer(self._te_layers, dev, pin=False)

    def _encode_prompt(self, task: str) -> None:
        """Fill the policy's text-context cache for `task` and the empty CFG prompt."""
        from lerobot.policies.flux3.f3 import packing

        policy = self.policy
        captions = {packing.select_caption(task, False), ""}
        missing = [c for c in captions if c not in policy._ctx_cache]
        if not missing:
            return
        te = policy.frozen.text_encoder.model
        layer_ids = {id(p) for p in self._te_layers.parameters()}
        rest = [p for p in te.parameters() if id(p) not in layer_ids] + list(te.buffers())
        t0 = time.perf_counter()
        for p in rest:
            p.data = p.data.to(self.device)
        try:
            for c in missing:
                policy._context(c, self.device)
        finally:
            for p in rest:
                p.data = p.data.to("cpu")
            self.torch.cuda.empty_cache()
        print(f"encoded {len(missing)} prompt(s) in {time.perf_counter() - t0:.1f} s: {missing}")

    def reset(self, task: str = "") -> None:
        self.policy.reset()
        self.pre.reset()
        self.post.reset()
        if self.offload and task:
            self._encode_prompt(task)

    def act(self, req: ActReq) -> tuple[list[float], float | None]:
        torch = self.torch

        def img(b64: str, key: str):
            image = torch.from_numpy(_decode(b64).copy()).permute(2, 0, 1).float().div(255.0)[None]
            # The saved side-by-side camera processor only aligns grid layouts.
            # Match each checkpoint input shape before it accumulates image history.
            hw = tuple(self.policy.config.input_features[key].shape[-2:])
            if tuple(image.shape[-2:]) != hw:
                image = torch.nn.functional.interpolate(
                    image, size=hw, mode="bilinear", align_corners=False, antialias=True)
            return image

        batch = {
            "observation.state": torch.tensor([req.state], dtype=torch.float32),
            "observation.images.scene": img(req.scene, "observation.images.scene"),
            "observation.images.wrist": img(req.wrist, "observation.images.wrist"),
            "task": req.task,
        }
        new_chunk = len(self.policy._action_queue) == 0
        if new_chunk and self.offload:
            self._encode_prompt(req.task)  # no-op unless the task changed without a /reset
        t0 = time.perf_counter()
        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(batch)))
        if new_chunk and self.device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 if new_chunk else None
        return action.reshape(-1).float().cpu().tolist(), ms


class FakeRunner:
    """Holds the arm still. Exercises the client, the protocol and the unit conversions without a GPU."""

    def reset(self, task: str = "") -> None:
        pass

    def act(self, req: ActReq) -> tuple[list[float], float | None]:
        _decode(req.scene), _decode(req.wrist)
        return list(req.state), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--offload", action="store_true",
                    help="stream the 7B transformer + text encoder through a small GPU (e.g. 8 GB); slower")
    args = ap.parse_args()

    runner = FakeRunner() if args.fake else FluxRunner(args.repo, args.revision, args.device, args.offload)
    lock = threading.Lock()  # one session, single-flight
    app = FastAPI()

    @app.post("/reset")
    def reset(req: ResetReq):
        with lock:
            runner.reset(req.task)
        return {"ok": True}

    @app.post("/act")
    def act(req: ActReq):
        if len(req.state) != 6:
            return {"error": "state must have 6 values"}
        with lock:
            action, ms = runner.act(req)
        return {"action": action, "infer_ms": ms}

    @app.get("/health")
    def health():
        return {"ok": True, "fake": args.fake, "model": args.repo,
                "revision": args.revision, "device": args.device,
                "action_dim": 6, "session_mode": "single-arm, reset between episodes"}

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
