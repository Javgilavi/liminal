"""Research/demo companion for an existing Haller controller. No motor connection on startup."""

import asyncio
import contextlib
import json
import secrets
import time
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from urllib.parse import urlsplit


def create_research_app(
    profile: str | None = None,
    controller: str = "http://127.0.0.1:8000",
    site_url: str | None = None,
):
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, Response
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    from liminal.ui.twin import JointMirror, parse_telemetry

    root = Path(__file__).resolve().parents[3]
    assets = Path(__file__).with_name("static") / "research"
    mirror = JointMirror(root / "examples/dual_so101_sim/research_replay/scene.xml")
    geometry = mirror.geometry()
    public_origin = None
    pairing_token = secrets.token_urlsafe(32)
    if site_url:
        site_address = urlsplit(site_url)
        if (
            site_address.scheme != "https"
            or not site_address.hostname
            or site_address.username
            or site_address.query
            or site_address.fragment
        ):
            raise ValueError("site-url must be an HTTPS site URL without credentials, query or fragment")
        public_origin = f"https://{site_address.netloc}"
        site_url = site_url.rstrip("/") + "/"
    controller = controller.rstrip("/")
    address = urlsplit(controller)
    if address.scheme not in ("http", "https") or not address.hostname or address.username:
        raise ValueError("controller must be an HTTP origin without credentials")
    ws_url = ("wss" if address.scheme == "https" else "ws") + "://" + address.netloc + "/ws/telemetry"
    state = {"arms": {}, "received_at": None, "rate_hz": 0.0, "error": "Controller not connected"}
    job = {"state": "idle", "task": None, "error": None, "output": None}
    task = None
    lock = asyncio.Lock()
    loopback_hosts = {"127.0.0.1", "localhost", "::1", "testserver"}
    stopped = asyncio.Event()
    cfg = None
    if profile:
        import yaml

        cfg = yaml.safe_load(Path(profile).read_text())
        if cfg.get("arm") not in ("left", "right"):
            raise ValueError("profile must select one arm")
        if controller != "http://127.0.0.1:8000":
            raise ValueError("current official runner requires Haller at http://127.0.0.1:8000")

    def snapshot():
        age = None if state["received_at"] is None else time.monotonic() - state["received_at"]
        connected = age is not None and age < 0.5 and not state["error"]
        return {
            "connected": bool(connected),
            "sample_age_ms": None if age is None else round(age * 1000),
            "observed_hz": round(state["rate_hz"], 1),
            "arms": state["arms"],
            "poses": mirror.poses(),
            "error": state["error"],
            "job": {k: v for k, v in job.items()},
            "profile_configured": cfg is not None,
            "active_arm": cfg.get("arm") if cfg else None,
            "checkpoint_label": Path(str(cfg.get("checkpoint", "configured profile"))).name if cfg else None,
            "object_tracking": False,
            "source": "measured_joints",
            "controller": controller,
        }

    async def observe():
        try:
            from websockets.asyncio.client import connect
        except ImportError:
            state["error"] = "Install the live extra to receive Haller telemetry"
            return
        while not stopped.is_set():
            try:
                async with connect(ws_url, open_timeout=2, close_timeout=1) as ws:
                    async for raw in ws:
                        message = json.loads(raw)
                        if "t_ms" in message and abs(time.time() - float(message["t_ms"]) / 1000) > 0.5:
                            raise ValueError("Controller timestamp is stale or clocks differ by >500 ms")
                        arms = parse_telemetry(message)
                        now = time.monotonic()
                        if state["received_at"] is not None:
                            interval = now - state["received_at"]
                            if 0 < interval < 1:
                                state["rate_hz"] = state["rate_hz"] * 0.8 + 0.2 / interval
                        mirror.apply({name: arm["positions"] for name, arm in arms.items()})
                        state.update(arms=arms, received_at=now, error=None)
            except asyncio.CancelledError:
                raise
            except Exception:
                state["error"] = "Waiting for Haller telemetry"
                await asyncio.sleep(2)

    async def cancel():
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    @asynccontextmanager
    async def lifespan(app):
        observer = asyncio.create_task(observe())
        try:
            yield
        finally:
            stopped.set()
            observer.cancel()
            await cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await observer

    app = FastAPI(title="Liminal · Embodied in the loop", lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.mount("/assets", StaticFiles(directory=assets), name="research-assets")
    app.state.telemetry = state
    app.state.mirror = mirror

    @app.middleware("http")
    async def paired_origin(request: Request, call_next):
        if request.url.hostname not in loopback_hosts:
            return Response("Use the loopback companion address", status_code=400)
        if request.headers.get("sec-fetch-site") == "cross-site" and not request.headers.get("origin"):
            return Response("Cross-site embedding is disabled", status_code=403)
        origin = request.headers.get("origin")
        external = origin and urlsplit(origin).netloc != request.headers.get("host")
        if external:
            if origin != public_origin:
                return Response("Origin is not paired", status_code=403)
            if request.method != "OPTIONS":
                auth = request.headers.get("authorization", "")
                if not secrets.compare_digest(auth, "Bearer " + pairing_token):
                    return Response("Pair this browser with the local companion", status_code=401)
        response = await call_next(request)
        if request.url.path.startswith(("/api/", "/pair")):
            response.headers["Cache-Control"] = "no-store"
        if origin == public_origin and request.method == "OPTIONS":
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    @app.get("/pair")
    def pair(request: Request):
        if site_url is None:
            raise HTTPException(409, "Restart the companion with --site-url https://javgilavi.github.io/liminal/")
        if request.url.hostname not in loopback_hosts or (
            request.client and request.client.host not in ("127.0.0.1", "::1", "testclient")
        ):
            raise HTTPException(403, "Pair from the robot-side machine using its loopback address")
        from urllib.parse import urlencode

        fragment = urlencode({"bridge": str(request.base_url).rstrip("/"), "token": pairing_token})
        link = escape(site_url + "live.html#" + fragment, quote=True)
        return HTMLResponse(
            '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<meta name="referrer" content="no-referrer"><title>Pair Liminal</title>'
            "<style>body{max-width:600px;margin:12vh auto;padding:24px;background:#f7f9f2;"
            "color:#29462d;font:17px/1.8 system-ui}a{display:inline-block;background:#29462d;"
            "color:white;padding:13px 22px;border-radius:5px;text-decoration:none}</style>"
            "<h1>Pair your Liminal lab.</h1><p>This private link connects the public lab to this "
            "local companion. Cameras and robot data stay between your browser and your machine. "
            "The pairing session expires when the companion restarts.</p>"
            f'<a href="{link}" rel="noreferrer">Open the paired full-screen lab ↗</a>'
            "<p>Keep this link private. Hardware actions still require explicit engagement in Haller.</p>",
            headers={
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'",
                "Referrer-Policy": "no-referrer",
                "X-Frame-Options": "DENY",
            },
        )

    @app.get("/")
    def index():
        return FileResponse(assets / "index.html")

    @app.get("/api/scene")
    def scene():
        return geometry

    @app.get("/api/state")
    def get_state():
        return snapshot()

    @app.websocket("/ws/twin")
    async def updates(websocket: WebSocket):
        if websocket.url.hostname not in loopback_hosts:
            await websocket.close(code=1008)
            return
        origin = websocket.headers.get("origin")
        external = origin and urlsplit(origin).netloc != websocket.headers.get("host")
        if external:
            protocols = websocket.scope.get("subprotocols", [])
            authorized = any(secrets.compare_digest(p, "token." + pairing_token) for p in protocols)
            if origin != public_origin or not authorized:
                await websocket.close(code=1008)
                return
        await websocket.accept(subprotocol="liminal" if external else None)
        try:
            while True:
                await websocket.send_json(snapshot())
                await asyncio.sleep(1 / 30)
        except WebSocketDisconnect:
            pass

    @app.get("/api/cameras/{name}")
    async def camera(name: str):
        if name not in ("top", "wrist_a", "wrist_b"):
            raise HTTPException(404, "Unknown camera")
        if not snapshot()["connected"]:
            raise HTTPException(503, "Real camera unavailable while controller is disconnected")
        try:
            import urllib.request

            def fetch():
                with urllib.request.urlopen(f"{controller}/cameras/{name}/snapshot", timeout=2) as response:
                    data = response.read(8 * 1024 * 1024 + 1)
                    if not response.headers.get("content-type", "").startswith("image/"):
                        raise ValueError("Controller did not return an image")
                    if len(data) > 8 * 1024 * 1024:
                        raise ValueError("Camera frame exceeds size limit")
                    return data

            return Response(
                await asyncio.to_thread(fetch), media_type="image/jpeg", headers={"Cache-Control": "no-store"}
            )
        except Exception as exc:
            raise HTTPException(503, "Controller camera unavailable") from exc

    class RunRequest(BaseModel):
        prompt: str = Field(min_length=1, max_length=500)
        execute: bool = False
        engage_acknowledged: bool = False
        duration_s: float = Field(default=20, ge=1, le=120)

    @app.post("/api/run")
    async def run(req: RunRequest):
        nonlocal task
        async with lock:
            if not req.prompt.strip():
                raise HTTPException(422, "An instruction is required")
            if cfg is None:
                raise HTTPException(409, "Start with --profile to connect the existing guarded FLUX runner")
            current = snapshot()
            if not current["connected"]:
                raise HTTPException(409, "Fresh controller telemetry is required")
            if task is not None and not task.done():
                raise HTTPException(409, "A trial is already running")
            if req.execute and not req.engage_acknowledged:
                raise HTTPException(409, "Acknowledge MANUAL/torque engagement in Haller before execution")
            arm = current["arms"].get(cfg["arm"], {})
            if arm.get("mode") == "policy":
                raise HTTPException(409, "Selected arm already belongs to another policy")
            if req.execute and (arm.get("mode") != "manual" or not arm.get("torque")):
                raise HTTPException(409, "Select MANUAL and engage the selected arm in Haller first")
            if req.execute and (cfg.get("camera_verified") is not True or cfg.get("units_verified") is not True):
                raise HTTPException(409, "Profile camera binding and units must be verified")
            output = root / "runs/research" / f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1000000:06d}"
            output.mkdir(parents=True, exist_ok=False)
            job.update(state="starting", task=req.prompt.strip(), error=None, output=str(output))

            async def go():
                try:
                    from liminal.policy.live import run as guarded_run

                    job["state"] = "running"
                    await guarded_run(
                        {**cfg, "prompt": req.prompt.strip(), "duration_s": req.duration_s}, output, req.execute
                    )
                    job["state"] = "finished"
                except asyncio.CancelledError:
                    job["state"] = "stopped"
                    raise
                except Exception as exc:
                    job.update(state="error", error=str(exc))

            task = asyncio.create_task(go())
            return {"started": True, "execute": req.execute, "output": str(output)}

    @app.post("/api/stop")
    async def stop():
        await cancel()  # existing runner's finally path requests Haller stop and torque release
        return {"stopped": True, "scope": "trial launched by this companion"}

    if public_origin:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[public_origin],
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )
    return app
