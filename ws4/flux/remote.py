"""Serve FLUX backends from the GPU box; call them from the robot computer.

LeRobot's remote inference does not support FLUX 3 Action: its policy table marks history and
previous-command conditioning as unsupported (lerobot@146d847:src/lerobot/remote_inference/README.md:161).
Our runner keeps the history on the robot side, so the server can stay stateless: one POST per chunk carries
the window (8 x D states and commands, plus only the encoded image snapshots, 2 x 256x256 per camera).

Transport is stdlib HTTP with ``.npz`` bodies (``allow_pickle=False``). There is no TLS; bind to 127.0.0.1 on
the GPU box and reach it through ``ssh -L``. An optional shared token is checked on every request.

GPU box:   python -m flux.remote --package so101=black-forest-labs/flux-3-action-so101 --port 8765
Robot PC:  ssh -N -L 8765:127.0.0.1:8765 user@gpu-box   then RemoteBackend("http://127.0.0.1:8765", "so101")

Imagination (opt-in, ws4/imagine/serving.py): ``RemoteBackend(..., imagine="sync" | "async" | "lazy")`` also gets
each chunk's imagined frames, in the response (``sync``) or later via ``fetch_imagination(id)`` (``GET
/imagination/<package>/<id>``). Actions are identical either way; requests without ``imagine`` are unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses
import hmac
import io
import json
import os
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from .backends import FluxBackend, PolicyRequest, PolicyResponse
from .contract import ModelContract

TOKEN_HEADER = "X-Flux-Token"
_IMG_PREFIX = "image::"
_IMAGINED_PREFIX = "imagined::"


def _pack(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(buf, **payload)
    return buf.getvalue()


def _unpack(body: bytes) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(io.BytesIO(body), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(arrays.pop("__meta__").tobytes().decode())
    return arrays, meta


def _jpeg_encode(frames: np.ndarray, quality: int) -> tuple[np.ndarray, list[int]]:
    from PIL import Image

    chunks = []
    for frame in frames:
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
        chunks.append(buf.getvalue())
    return np.frombuffer(b"".join(chunks), dtype=np.uint8), [len(c) for c in chunks]


def _jpeg_decode(data: np.ndarray, lengths: list[int]) -> np.ndarray:
    from PIL import Image

    raw, frames, start = data.tobytes(), [], 0
    for n in lengths:
        frames.append(np.asarray(Image.open(io.BytesIO(raw[start : start + n])).convert("RGB"), dtype=np.uint8))
        start += n
    return np.stack(frames)


def encode_request(request: PolicyRequest, codec: str = "jpeg", quality: int = 95) -> bytes:
    """``codec="jpeg"`` sends ~20 KB per 256x256 frame instead of 192 KB raw. The SO-101 corpus itself was AV1
    video (lerobot community datasets), so mild compression stays within what the model saw; use ``raw`` to
    rule it out."""
    arrays = {"states": request.states, "commands": request.commands}
    meta: dict[str, Any] = {
        "task": request.task,
        "snapshot_ticks": list(request.snapshot_ticks),
        "target": request.target,
        "codec": codec,
        "lengths": {},
    }
    if request.imagine:  # absent by default, so plain requests are byte-identical to before
        meta["imagine"] = request.imagine
    for key, frames in request.images.items():
        if codec == "jpeg":
            arrays[_IMG_PREFIX + key], meta["lengths"][key] = _jpeg_encode(frames, quality)
        elif codec == "raw":
            arrays[_IMG_PREFIX + key] = frames
        else:
            raise ValueError("codec must be jpeg or raw")
    return _pack(arrays, meta)


def decode_request(body: bytes) -> PolicyRequest:
    arrays, meta = _unpack(body)
    images = {k.removeprefix(_IMG_PREFIX): v for k, v in arrays.items() if k.startswith(_IMG_PREFIX)}
    if meta.get("codec", "raw") == "jpeg":
        images = {k: _jpeg_decode(v, meta["lengths"][k]) for k, v in images.items()}
    return PolicyRequest(
        task=str(meta["task"]),
        states=arrays["states"].astype(np.float64),
        commands=arrays["commands"].astype(np.float64),
        images=images,
        snapshot_ticks=tuple(int(t) for t in meta["snapshot_ticks"]),
        target=str(meta.get("target", "")),
        imagine=str(meta.get("imagine", "")),
    )


def _imagination_body(imagination: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {_IMAGINED_PREFIX + k: v for k, v in imagination["arrays"].items()}
    return arrays, imagination["meta"]


def _imagination_from_body(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> dict[str, Any]:
    return {"meta": meta, "arrays": {k.removeprefix(_IMAGINED_PREFIX): v for k, v in arrays.items()
                                     if k.startswith(_IMAGINED_PREFIX)}}  # fmt: skip


def contract_to_json(contract: ModelContract) -> dict[str, Any]:
    return dataclasses.asdict(contract)


def contract_from_json(data: dict[str, Any]) -> ModelContract:
    data = dict(data)
    for key in ("camera_keys", "image_hw", "absolute_dims"):
        data[key] = tuple(data[key])
    return ModelContract(**data)


def make_server(backends: dict[str, FluxBackend], host: str = "127.0.0.1", port: int = 8765, token: str | None = None):
    """A ``ThreadingHTTPServer``; each backend serializes its own calls (one lock per loaded model)."""

    class Handler(BaseHTTPRequestHandler):
        def _authorized(self) -> bool:
            if token is None:
                return True
            return hmac.compare_digest(self.headers.get(TOKEN_HEADER, ""), token)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code: int, message: str) -> None:
            self._send(code, json.dumps({"error": message}).encode(), "application/json")

        def _backend(self, prefix: str) -> tuple[str, FluxBackend] | None:
            name = self.path.removeprefix(prefix).strip("/")
            if name not in backends:
                self._error(404, f"unknown package {name!r}; have {sorted(backends)}")
                return None
            return name, backends[name]

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if not self._authorized():
                return self._error(401, "bad token")
            if self.path.rstrip("/") == "/packages":
                return self._send(200, json.dumps(sorted(backends)).encode(), "application/json")
            if self.path.startswith("/imagination/"):
                return self._imagination()
            found = self._backend("/contract")
            if found is not None and self.path.startswith("/contract"):
                self._send(200, json.dumps(contract_to_json(found[1].contract)).encode(), "application/json")

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return self._error(401, "bad token")
            if not self.path.startswith("/predict"):
                return self._error(404, "POST /predict/<package>")
            found = self._backend("/predict")
            if found is None:
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                request = decode_request(self.rfile.read(length))
                response = found[1].predict(request)
            except Exception as exc:  # report to the client instead of dropping the connection
                return self._error(400, f"{type(exc).__name__}: {exc}")
            arrays: dict[str, np.ndarray] = {"commands": response.commands}
            meta: dict[str, Any] = {"latency_s": response.latency_s, "info": response.info}
            if response.imagination is not None:
                extra, meta["imagination"] = _imagination_body(response.imagination)
                arrays.update(extra)
            self._send(200, _pack(arrays, meta), "application/octet-stream")

        def _imagination(self) -> None:
            """GET /imagination/<package>/<id>[?wait=seconds]"""
            from urllib.parse import parse_qs, urlparse

            url = urlparse(self.path)
            parts = url.path.strip("/").split("/")
            if len(parts) != 3 or parts[1] not in backends:
                return self._error(404, "GET /imagination/<package>/<id>")
            fetch = getattr(backends[parts[1]], "fetch_imagination", None)
            if fetch is None:
                return self._error(404, f"package {parts[1]!r} has no imagination")
            wait_s = float(parse_qs(url.query).get("wait", ["30"])[0])
            try:
                meta, arrays = fetch(parts[2], wait_s)
            except KeyError as exc:
                return self._error(404, str(exc))
            except TimeoutError as exc:
                return self._error(504, str(exc))
            except Exception as exc:
                return self._error(500, f"{type(exc).__name__}: {exc}")
            body_arrays, body_meta = _imagination_body({"meta": meta, "arrays": arrays})
            self._send(200, _pack(body_arrays, {"imagination": body_meta}), "application/octet-stream")

        def log_message(self, fmt: str, *args: Any) -> None:  # keep the control loop's terminal quiet
            pass

    return ThreadingHTTPServer((host, port), Handler)


class RemoteBackend:
    """Client side of :func:`make_server`, usable wherever a ``FluxBackend`` is expected."""

    def __init__(
        self,
        url: str,
        package: str = "so101",
        token: str | None = None,
        timeout_s: float = 30.0,
        codec: str = "jpeg",
        quality: int = 95,
        imagine: str = "",
    ):
        self.url = url.rstrip("/")
        self.package = package
        self.token = token
        self.timeout_s = timeout_s
        self.codec, self.quality = codec, quality
        self.imagine = imagine  # "", "sync", "async" or "lazy" (ws4/imagine/serving.py)
        self.contract = contract_from_json(json.loads(self._call("GET", f"/contract/{package}")))

    def _call(self, method: str, path: str, body: bytes | None = None) -> bytes:
        req = urllib.request.Request(self.url + path, data=body, method=method)
        if self.token is not None:
            req.add_header(TOKEN_HEADER, self.token)
        if body is not None:
            req.add_header("Content-Type", "application/octet-stream")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"policy server {exc.code}: {exc.read().decode(errors='replace')}") from exc

    def predict(self, request: PolicyRequest) -> PolicyResponse:
        start = time.perf_counter()
        if self.imagine and not request.imagine:
            request = dataclasses.replace(request, imagine=self.imagine)
        body = encode_request(request, self.codec, self.quality)
        arrays, meta = _unpack(self._call("POST", f"/predict/{self.package}", body))
        meta.setdefault("info", {})["request_bytes"] = len(body)
        info = dict(meta["info"])
        info["server_latency_s"] = meta.get("latency_s")
        imagination = None
        if "imagination" in meta:
            imagination = _imagination_from_body(arrays, meta["imagination"])
            info["imagination_id"] = meta["imagination"].get("id")
        return PolicyResponse(arrays["commands"].astype(np.float64), time.perf_counter() - start, info, imagination)

    def fetch_imagination(self, imagination_id: str, wait_s: float = 60.0) -> dict[str, Any]:
        """``{"meta", "arrays"}`` of an ``async`` / ``lazy`` imagination (the server decodes it if needed);
        ``imagine.serving.imagination_from_payload(**...)`` turns it into frames."""
        body = self._call("GET", f"/imagination/{self.package}/{imagination_id}?wait={wait_s:g}")
        arrays, meta = _unpack(body)
        return _imagination_from_body(arrays, meta["imagination"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve LeRobot flux3 packages to the robot computer.")
    parser.add_argument(
        "--package",
        action="append",
        required=True,
        metavar="NAME=CHECKPOINT[@DEVICE]",
        help="repeatable; packages with the same CHECKPOINT and DEVICE share one loaded model",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default=None, help="prefer --token-file or $FLUX_TOKEN: argv is visible in ps")
    parser.add_argument("--token-file", default=None, help="file holding the shared token")
    parser.add_argument("--text-encoder-on-cpu", action="store_true")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=JSON", help="Flux3Config field")
    args = parser.parse_args(argv)
    from .backends import LeRobotFlux3Backend

    overrides = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.override)}
    loaded: dict[tuple[str, str], LeRobotFlux3Backend] = {}
    backends: dict[str, FluxBackend] = {}
    for item in args.package:
        name, spec = item.split("=", 1)
        checkpoint, _, device = spec.partition("@")
        key = (checkpoint, device or "cuda")
        if key not in loaded:
            loaded[key] = LeRobotFlux3Backend.from_pretrained(
                checkpoint, device=key[1], overrides=overrides, text_encoder_on_cpu=args.text_encoder_on_cpu
            )
        backends[name] = loaded[key]
        print(f"{name}: {checkpoint} on {key[1]} -> {loaded[key].contract}")
    token = args.token or os.environ.get("FLUX_TOKEN")
    if args.token_file:
        with open(args.token_file) as f:
            token = f.read().strip()
    server = make_server(backends, args.host, args.port, token)
    print(f"serving {sorted(backends)} on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
