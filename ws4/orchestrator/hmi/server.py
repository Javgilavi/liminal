"""Localhost HMI: stdlib HTTP server, one static page, JSON API over the session.

Bound to 127.0.0.1 only. Requests whose Host header is not localhost are refused (DNS rebinding), and every POST
must carry ``X-HMI: 1`` with a JSON body: a custom header forces a CORS preflight that this server never approves, so
another web page in the operator's browser cannot approve a plan or move the arms.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..plan import plan_json_schema

PAGE = Path(__file__).resolve().parent / "index.html"
LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")


def make_server(session: Any, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "orchestrator-hmi"

        def _host_ok(self) -> bool:
            host = self.headers.get("Host") or ""
            name = host[: host.find("]") + 1] if host.startswith("[") else host.split(":")[0]
            return name in LOCAL_HOSTS

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, data: Any) -> None:
            self._send(code, json.dumps(data, default=str).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if not self._host_ok():
                return self._json(403, {"error": "localhost only"})
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._json(200, session.snapshot())
            if path == "/api/schema":
                return self._json(200, plan_json_schema(session.vocab))
            if path.startswith("/api/frame/") and path.endswith(".jpg"):
                frame = session.frame(path[len("/api/frame/"):-4])
                if frame is None:
                    return self._json(404, {"error": "no frame"})
                return self._send(200, frame, "image/jpeg")
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._host_ok():
                return self._json(403, {"error": "localhost only"})
            if self.headers.get("X-HMI") != "1" or "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._json(403, {"error": "POST needs X-HMI: 1 and a JSON body"})
            length = min(int(self.headers.get("Content-Length", "0") or 0), 1_000_000)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "bad JSON"})
            path = self.path.split("?", 1)[0]
            if path == "/api/estop":
                session.estop("HMI")
                return self._json(200, {"ok": True})
            if path == "/api/release":
                session.release()
                return self._json(200, {"ok": True})
            if path == "/api/resume":
                session.resume()
                return self._json(200, {"ok": True})
            if path == "/api/abort":
                session.abort()
                return self._json(200, {"ok": True})
            if path == "/api/plan":
                goal = str(body.get("goal", "")).strip()
                if not goal:
                    return self._json(400, {"error": "empty goal"})
                if session.planning:
                    return self._json(409, {"error": "already planning"})
                run = session.executor
                if run is not None and run.state not in ("succeeded", "failed", "aborted"):
                    return self._json(409, {"error": "a run is in progress"})
                threading.Thread(target=session.plan, args=(goal,), daemon=True, name="plan").start()
                return self._json(202, {"ok": True})
            if path == "/api/plan_json":
                proposal = session.load_plan_json(body.get("plan", {}), str(body.get("goal", "")))
                return self._json(200, proposal.to_json())
            if path == "/api/approve":
                msg = session.approve(int(body.get("version", -1)))
                return self._json(200 if msg == "ok" else 409, {"ok": msg == "ok", "message": msg})
            if path == "/api/confirm":
                session.confirm(str(body.get("step", "")))
                return self._json(200, {"ok": True})
            if path == "/api/override":
                session.override(str(body.get("step", "")), bool(body.get("done")))
                return self._json(200, {"ok": True})
            if path == "/api/reset":
                seed = body.get("seed")
                msg = session.reset_scene(None if seed in (None, "") else int(seed), body.get("lid_on_box"))
                return self._json(200 if msg == "ok" else 409, {"ok": msg == "ok", "message": msg})
            self._json(404, {"error": "not found"})

        def log_message(self, fmt: str, *args: Any) -> None:  # keep the terminal for events
            pass

    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("the HMI binds to localhost only")
    return ThreadingHTTPServer((host, port), Handler)
