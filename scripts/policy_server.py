"""Policy-agnostic LeRobot inference server for the Haller sim (and liminal's HttpPolicyBackend).

Speaks the same protocol as ``dual_so101_sim/flux/server.py``, so ``flux/app.py`` and liminal both work against
either server. FLUX 3 Action stays on that server (its processors integrate command deltas and keep their own
history). This one loads any LeRobot policy that predicts absolute joint targets in chunks (SmolVLA, pi0.5, ACT, ...)
by repo id.

    # separate venv, never liminal's: uv venv .venv-policy; torch cu128; pip install -e 'lerobot[smolvla,pi]'
    ../dual_so101_sim/.venv-policy/bin/python scripts/policy_server.py --policy lerobot/smolvla_base --port 8100
    ../dual_so101_sim/.venv-policy/bin/python scripts/policy_server.py --policy victorvanhalst/smolvla_so101_cube
    ../dual_so101_sim/.venv-policy/bin/python scripts/policy_server.py \
        --policy 00ri/so101_megamix_v1_pi05 --dtype bfloat16
    ../dual_so101_sim/.venv-policy/bin/python scripts/policy_server.py --fake      # echoes the state, no model

Protocol (JSON), a superset of flux/server.py:
    POST /reset {"task": str, "session": str = "default"}                -> {"ok": true}
    POST /act   {"task": str, "state": [6], "scene": b64 JPEG, "wrist": b64 JPEG,
                 "images": {role: b64 JPEG} = {}, "session": str = "default"}
                -> {"action": [6], "infer_ms": float | null, "chunk": int}
    GET  /health -> policy, type, camera map, chunk sizes, fake
``session`` keeps one action queue per caller (liminal uses one per arm, AGENTS.md rule 6); ``infer_ms`` is set only
on ticks that ran the model (one per chunk). State/action units: degrees for the five arm joints, 0-100 for the
gripper (LeRobot SO-101 convention, zero = mid-range), unless ``--units m100``.
"""

from __future__ import annotations

import argparse
import base64
import io
import math
import threading
import time
from collections import deque
from typing import Any

# Protocol role -> checkpoint image key, for checkpoints whose camera names don't say what they are.
# VERIFY: every entry is read off the checkpoint's config.json, not confirmed against its training data.
CAMERA_PRESETS: dict[str, dict[str, str]] = {
    # SmolVLA pretraining renamed community SO-100 cameras to camera1..3. Assumed camera1 = top/scene view,
    # camera2 = wrist, camera3 = side (absent here, so it is dropped: empty_cameras = 0). VERIFY.
    "lerobot/smolvla_base": {"scene": "observation.images.camera1", "wrist": "observation.images.camera2"},
    # SO-101 cube pick-and-place fine-tune of smolvla_base, same camera naming. VERIFY.
    "victorvanhalst/smolvla_so101_cube": {"scene": "observation.images.camera1",
                                          "wrist": "observation.images.camera2"},
    # pi0.5 fine-tuned on a mix of SO-101 community datasets: top / gripper / front at 480x640. VERIFY.
    "00ri/so101_megamix_v1_pi05": {"scene": "observation.images.top", "wrist": "observation.images.gripper",
                                   "front": "observation.images.front"},
}
# Name fragments for checkpoints with descriptive camera keys (first match wins, in this order).
ROLE_HINTS = (("wrist", ("wrist", "gripper", "hand")), ("scene", ("top", "up", "scene", "base", "overhead")),
              ("front", ("front", "side")))

# SO-101 MJCF joint half-ranges in degrees (dual_so101_sim/so101/so101.xml), used only for --units m100, i.e.
# checkpoints recorded in LeRobot's default RANGE_M100_100 mode. The real arm's calibrated range differs (README:
# "~5-10% short"), so this mapping is approximate. VERIFY per checkpoint.
SO101_HALF_RANGE_DEG = (110.0, 100.0, 96.8, 95.0, 160.0)


def decode_jpeg(b64: str) -> Any:
    import numpy as np
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def to_units(state: list[float], units: str) -> list[float]:
    if units == "degrees":
        return list(state)
    return [v / h * 100.0 for v, h in zip(state[:5], SO101_HALF_RANGE_DEG, strict=True)] + list(state[5:])


def from_units(action: list[float], units: str) -> list[float]:
    if units == "degrees":
        return list(action)
    return [v * h / 100.0 for v, h in zip(action[:5], SO101_HALF_RANGE_DEG, strict=True)] + list(action[5:])


def camera_map(repo: str, image_keys: list[str], overrides: dict[str, str]) -> dict[str, str]:
    """Protocol role -> checkpoint image key: explicit --camera, then a preset, then name hints."""
    if overrides:
        unknown = set(overrides.values()) - set(image_keys)
        if unknown:
            raise SystemExit(f"--camera names unknown image keys {sorted(unknown)}; checkpoint has {image_keys}")
        return dict(overrides)
    if repo in CAMERA_PRESETS:
        return {r: k for r, k in CAMERA_PRESETS[repo].items() if k in image_keys}
    mapping: dict[str, str] = {}
    for key in image_keys:
        name = key.rsplit(".", 1)[-1].lower()
        for role, hints in ROLE_HINTS:
            if role not in mapping and any(h in name for h in hints):
                mapping[role] = key
                break
    if not mapping:
        raise SystemExit(f"cannot guess which camera is which in {image_keys}; pass --camera scene=KEY wrist=KEY")
    return mapping


class Session:
    """One caller's action queue. Chunk boundaries are per session, so two arms never share a chunk."""

    def __init__(self, task: str = ""):
        self.task = task
        self.queue: deque[list[float]] = deque()
        self.chunks = 0


class LeRobotRunner:
    def __init__(self, repo: str, revision: str | None, device: str, dtype: str | None, cameras: dict[str, str],
                 units: str, n_action_steps: int | None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        self.device, self.units = device, units
        cfg = PreTrainedConfig.from_pretrained(repo, revision=revision)
        cfg.device = device
        if dtype and hasattr(cfg, "dtype"):
            # pi0.5 declares `dtype` (float32 default); bfloat16 halves VRAM (3.6B params -> ~7.3 GB).
            cfg.dtype = getattr(torch, dtype) if not isinstance(cfg.dtype, str) else dtype
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(repo, revision=revision, config=cfg)
        self.policy.to(device)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=repo, pretrained_revision=revision,
            preprocessor_overrides={"device_processor": {"device": device}})
        self.config = self.policy.config
        image_keys = list(self.config.image_features)
        self.cameras = camera_map(repo, image_keys, cameras)
        self.state_dim = self.config.robot_state_feature.shape[0] if self.config.robot_state_feature else None
        self.action_dim = self.config.action_feature.shape[0]
        self.n_action_steps = min(n_action_steps or self.config.n_action_steps, self.config.chunk_size)
        self.info = {"policy": repo, "revision": revision, "type": cfg.type, "image_keys": image_keys,
                     "cameras": self.cameras, "chunk_size": self.config.chunk_size,
                     "n_action_steps": self.n_action_steps, "state_dim": self.state_dim,
                     "action_dim": self.action_dim, "units": units, "device": device}
        print(f"loaded {self.info}")
        if device.startswith("cuda"):
            print(f"GPU memory in use after load: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    def reset(self, session: Session) -> None:
        session.queue.clear()
        self.policy.reset()  # observation history; n_obs_steps is 1 for SmolVLA / pi0.5

    def act(self, session: Session, task: str, state: list[float], images: dict[str, str]) -> tuple[list[float],
                                                                                                     float | None]:
        torch = self.torch
        ms = None
        if not session.queue:
            batch: dict[str, Any] = {"observation.state": torch.tensor([to_units(state, self.units)],
                                                                        dtype=torch.float32), "task": task}
            for role, key in self.cameras.items():
                if role in images:
                    img = decode_jpeg(images[role])
                    batch[key] = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)[None]
            if not any(k in batch for k in self.cameras.values()):
                raise ValueError(f"no image for any of the checkpoint's cameras {self.cameras}")
            t0 = time.perf_counter()
            with torch.inference_mode():
                chunk = self.policy.predict_action_chunk(self.pre(batch))[:, : self.n_action_steps]
                actions = self.post(chunk)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            rows = actions.reshape(-1, actions.shape[-1]).float().cpu().tolist()
            session.queue.extend(from_units(r[: self.action_dim], self.units) for r in rows)
            session.chunks += 1
        return session.queue.popleft(), ms


class FakeRunner:
    """Holds the arm still: exercises the client, the protocol and the unit conversions without a GPU."""

    info = {"policy": "fake", "type": "fake", "cameras": {}, "chunk_size": 1, "n_action_steps": 1}

    def reset(self, session: Session) -> None:
        session.queue.clear()

    def act(self, session: Session, task: str, state: list[float], images: dict[str, str]) -> tuple[list[float],
                                                                                                     float | None]:
        for b64 in images.values():
            decode_jpeg(b64)
        return list(state), None


def main() -> None:
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", default="lerobot/smolvla_base", help="LeRobot policy repo id or local path")
    ap.add_argument("--revision", default=None, help="pin the HF commit")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None, help="e.g. bfloat16 for pi0.5 on an 8 GB GPU")
    ap.add_argument("--camera", nargs="*", default=[], metavar="ROLE=KEY",
                    help="map protocol roles (scene, wrist, front) to checkpoint image keys")
    ap.add_argument("--units", default="degrees", choices=["degrees", "m100"],
                    help="joint units the checkpoint was trained in (VERIFY per checkpoint)")
    ap.add_argument("--n-action-steps", type=int, default=None, help="actions executed per chunk (<= chunk_size)")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    overrides = dict(item.split("=", 1) for item in args.camera)
    runner: Any = FakeRunner() if args.fake else LeRobotRunner(
        args.policy, args.revision, args.device, args.dtype, overrides, args.units, args.n_action_steps)
    sessions: dict[str, Session] = {}
    lock = threading.Lock()  # one model, single-flight

    class ResetReq(BaseModel):
        task: str = ""
        session: str = "default"

    class ActReq(BaseModel):
        task: str
        state: list[float]
        scene: str | None = None
        wrist: str | None = None
        images: dict[str, str] = {}
        session: str = "default"

    app = FastAPI()

    @app.post("/reset")
    def reset(req: ResetReq):
        with lock:
            session = sessions[req.session] = Session(req.task)
            runner.reset(session)
        return {"ok": True}

    @app.post("/act")
    def act(req: ActReq):
        if len(req.state) != 6 or not all(math.isfinite(v) for v in req.state):
            return {"error": "state must have 6 finite values"}
        images = {**req.images, **{k: v for k, v in (("scene", req.scene), ("wrist", req.wrist)) if v}}
        with lock:
            session = sessions.setdefault(req.session, Session(req.task))
            if session.task != req.task:  # a new instruction without /reset: drop the stale chunk
                session.task = req.task
                runner.reset(session)
            try:
                action, ms = runner.act(session, req.task, req.state, images)
            except ValueError as e:
                return {"error": str(e)}
        return {"action": action, "infer_ms": ms, "chunk": session.chunks}

    @app.get("/health")
    def health():
        return {"ok": True, "fake": args.fake, **runner.info, "sessions": sorted(sessions)}

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
