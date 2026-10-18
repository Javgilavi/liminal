"""Run the haller_ws HMI backend with the FLUX recording settings, without editing haller_ws.

The only thing a config file cannot set is the recorder's video codec: haller_ws hard-codes
`DatasetRecorder.vcodec = "h264"` (hmi/backend/haller_hmi/recorder.py, dataclass field) and `server.py`
constructs the recorder without passing it. This launcher swaps `haller_hmi.server.DatasetRecorder` for a
partial with `vcodec` bound, then serves the unchanged app. Everything else comes from the YAML config.

Run it with the haller-hmi venv's python (that is where `haller_hmi` and LeRobot 0.5.1 live):

    ~/venvs/haller-hmi/bin/python ws4/teleop/backend.py --config ws4/teleop/configs/station-sim.yaml

Then `~/haller_ws/scripts/quest-teleop/up.sh --sim` (or `--local`) finds the backend already up on :8000 and
only adds the frontend and the HTTPS origin. Stop with SIGTERM (ws4/teleop/down.sh) and let it finish: the
lifespan shutdown is what finalises the dataset.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

RUN_DIR = Path(os.environ.get("WS4_TELEOP_RUN_DIR", "/tmp/ws4-teleop"))
# LeRobot 0.5.1 datasets/video_utils.py VALID_VIDEO_CODECS; "libsvtav1" is LeRobot's own default and what the
# public XLeRobot core set was written with.
CODECS = ("libsvtav1", "h264")
# The recorder writes fps = round(measured tick rate) and refuses a take when the measurement sits more than 0.5%
# from that integer (recorder.py FPS_FAITHFUL_FRACTION). The session loop sleeps `period - elapsed`, so each tick
# overshoots by the sleep latency: on this laptop (sim rig, AV1 recording) a 30.0 target measured 29.86 to 29.88 Hz
# at arm time (0.4 to 0.46% slow) and 30.12 measured 30.06. 30.08 centres it on 30. Re-measure on the real rig:
# validate.py prints `measured_hz` from info.json's haller_rate block.
SESSION_HZ = 30.08


def patch_recorder_codec(server_module, vcodec: str) -> None:
    """Make every DatasetRecorder the server builds use `vcodec`. Idempotent."""
    cls = getattr(server_module.DatasetRecorder, "func", server_module.DatasetRecorder)
    server_module.DatasetRecorder = functools.partial(cls, vcodec=vcodec)


def patch_session_rate(server_module, session_hz: float | None) -> None:
    """Force the headset session's control rate, whatever the client asks for.

    The recorder freezes `fps` from the session's tick bus, which publishes one sample per control tick
    (haller_hmi/human_teleop.py: `sample_hz` is None in server.py, so `_sample_divisor` is 1). The headset page
    starts sessions without `hz` (frontend components/VRTeleopPanel.tsx), so the default 60 Hz becomes a 60 fps
    dataset. `_hz_override` is the session's own constructor knob for exactly this (`effective_hz`).
    """
    server_module.human_teleop._hz_override = session_hz


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True, help="HMI YAML config (ws4/teleop/configs/*.yaml)")
    ap.add_argument("--vcodec", choices=CODECS, default="libsvtav1")
    ap.add_argument("--session-hz", type=float, default=SESSION_HZ,
                    help="headset session control rate (sets the recorded fps); 0 keeps the client's")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("HALLER_BACKEND_PORT", "8000")))
    args = ap.parse_args(argv)

    cfg = args.config.resolve()
    if not cfg.is_file():
        ap.error(f"no config at {cfg}")
    os.environ["HALLER_HMI_CONFIG"] = str(cfg)
    os.environ.setdefault("MUJOCO_GL", "egl")

    import haller_hmi.server as server
    import uvicorn  # haller-hmi venv

    patch_recorder_codec(server, args.vcodec)
    patch_session_rate(server, args.session_hz or None)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    stamp = {"pid": os.getpid(), "config": str(cfg), "vcodec": args.vcodec, "session_hz": args.session_hz,
             "port": args.port, "started_unix_s": time.time(), "python": sys.executable}
    (RUN_DIR / "backend.json").write_text(json.dumps(stamp, indent=1))
    print(f"[ws4-teleop] backend pid {os.getpid()} config={cfg.name} vcodec={args.vcodec} "
          f"session_hz={args.session_hz} port={args.port}",
          flush=True)
    # 10 s graceful shutdown, as haller_ws up.sh does: a headset parked on /teleop/vr holds a websocket forever,
    # and the lifespan shutdown behind it is what writes the parquet footers and meta/episodes.
    uvicorn.run(server.app, host=args.host, port=args.port, timeout_graceful_shutdown=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
