"""Operator HTTP API + web console (owned by WS5).

Legacy calls (kept): POST /run, POST /estop, GET /state.

Console API (replay/mock only, never hardware):
    GET  /api/options                      scenarios, tasks, demos, models, modes (+ memory if the service offers it)
    POST /api/ground        {scenario, goal, model}      -> proposal + per-arm atomic skill rows
    POST /api/ground_video  {scenario, demo, model} or a raw video body (?filename=) -> same (501 until wired)
    POST /api/runs          {scenario, mode, model, procedure, ...} -> {run_id}; the server re-validates
    GET  /api/runs/{id}                    status, report, obligations, attempts
    GET  /api/runs/{id}/events?since=N     polling; add &stream=1 for SSE
    GET  /api/runs/{id}/scene.png          current replay scene
    GET  /api/runs/{id}/memory             obligations, attempts, selected keyframes, experience
    GET  /api/frames/{run}/{frame_id}      one stored frame image
    POST /api/runs/{id}/resume

No `from __future__ import annotations` here: FastAPI resolves handler annotations at runtime and the
handlers are closures inside `create_app`.
"""

import asyncio
import dataclasses
import json
import re
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

WEB_DIR = Path(__file__).parent / "web"
RUNS_DIR = Path("runs")
RUN_FIELDS = ("scenario", "mode", "model", "goal", "task", "interrupt_after", "fail_first", "lie_first",
              "max_attempts", "max_model_calls", "memory")
BANNER = "REPLAY/MOCK, not hardware"


def _plain(value: Any) -> Any:
    """Make dataclasses/tuples/bytes JSON-safe."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class _Run:
    def __init__(self, run_id: str, prepared: Any, db: str | None, options: Any) -> None:
        self.run_id = run_id
        self.prepared = prepared
        self.db = db
        self.options = options
        self.task: asyncio.Task | None = None
        self.status = "prepared"
        self.report: Any = None
        self.error: str | None = None
        self.started = time.time()
        self.warnings: list[str] = []

    @property
    def active(self) -> bool:
        return self.task is not None and not self.task.done()

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.prepared.ctx.world.events


def create_app(robot: str, backend: str = "mock", policy: str | None = None, task_dir: str = "configs/tasks",
               runs_dir: str | Path | None = None):
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

    from liminal.app import build_context
    from liminal.orchestration import run_task, service
    from liminal.orchestration.planner import TemplatePlanner

    app = FastAPI(title="liminal")
    base = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    ctx = build_context(robot, backend=backend, policy=policy)
    planner = TemplatePlanner(task_dir)
    state: dict[str, Any] = {"running": None, "last_report": None}
    runs: dict[str, _Run] = {}
    current: dict[str, _Run | None] = {"run": None}

    def bad(message: str, code: int = 422) -> HTTPException:
        return HTTPException(code, message)

    def get_run(run_id: str) -> _Run:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(404, f"unknown run {run_id!r}")
        return run

    def known_path(value: Any, kind: str) -> str:
        """Scenario/task/demo names must come from the option lists: no arbitrary file reads."""
        listed = service.list_options().get(kind, [])
        if value not in listed:
            raise bad(f"unknown {kind[:-1]} {value!r}")
        return str(value)

    async def _cancel_after_grace(r: _Run, grace_s: float = 3.0) -> None:
        # Cancelling mid-planning makes run_episode close its SQLite ledger while the planner thread is still
        # inside it (segfault). The e-stop flag already halted the arms, so let the run unwind first.
        try:
            await asyncio.wait_for(asyncio.shield(r.task), grace_s)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
            if r.active:
                r.task.cancel()

    # -- legacy -------------------------------------------------------------------------------

    @app.on_event("startup")
    def _connect() -> None:
        for a in ctx.arms.values():
            a.connect()

    @app.post("/run")
    async def run(req: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        if state["running"] is not None and not state["running"].done():
            raise HTTPException(409, "a task is already running")
        try:
            spec = planner.plan(req.get("task") or req.get("goal") or "")
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        ctx.safety.reset_estop()

        async def _go() -> None:
            report = await run_task(spec, ctx)
            state["last_report"] = report.__dict__

        state["running"] = asyncio.create_task(_go())
        return {"started": spec.name}

    @app.post("/estop")
    async def estop() -> dict[str, Any]:
        ctx.safety.estop()
        if state["running"] is not None:
            state["running"].cancel()
        stopped = []
        for r in runs.values():
            if r.active:
                r.prepared.ctx.safety.estop()  # motors stop now
                r.status = "estopped"
                stopped.append(r.run_id)
                asyncio.create_task(_cancel_after_grace(r))
        return {"estopped": True, "runs": stopped}

    @app.get("/state")
    async def get_state() -> dict[str, Any]:
        out = {
            "facts": ctx.world.snapshot(),
            "owners": {a: ctx.safety.owner_of(a) for a in ctx.arms},
            "estopped": ctx.safety.estopped,
            "running": state["running"] is not None and not state["running"].done(),
            "last_report": state["last_report"],
            "events": ctx.world.events[-50:],
        }
        r = current["run"]
        if r is not None:
            out["console_run"] = {"run_id": r.run_id, "status": r.status, "active": r.active}
        return out

    # -- console: options / grounding ----------------------------------------------------------

    @app.get("/api/options")
    async def options() -> dict[str, Any]:
        opts = dict(service.list_options())
        opts["banner"] = BANNER
        opts["features"] = {"video": hasattr(service, "ground_demo"), "memory_options": "memory" in opts,
                            "inspect_run": _inspect_run() is not None}
        return _plain(opts)

    def _flatten(proposal: Any, rows: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
        d = proposal.to_dict()
        return {"proposal": d, "rows": rows, "warnings": list(d["unsupported"]) + list(d["unresolved"]), **extra}

    @app.post("/api/ground")
    async def ground(req: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        goal = str(req.get("goal") or "").strip()
        if not goal:
            raise bad("goal is empty")
        scenario = known_path(req.get("scenario"), "scenarios")
        try:
            proposal, rows = await asyncio.to_thread(service.ground, scenario, goal, req.get("model") or "replay")
        except (ValueError, KeyError) as e:  # ProcedureError is a ValueError
            raise bad(str(e)) from e
        except (RuntimeError, ImportError, OSError) as e:
            raise bad(f"grounding failed: {e}") from e
        return _flatten(proposal, rows)

    async def _demo_arg(request: Request) -> tuple[dict[str, Any], str]:
        if "application/json" in request.headers.get("content-type", ""):
            body = await request.json()
            return body, known_path(body.get("demo"), "demos")
        data = await request.body()  # raw upload: no multipart dependency
        if not data:
            raise bad("empty upload")
        name = re.sub(r"[^A-Za-z0-9._-]", "_", request.query_params.get("filename") or "demo.mp4")[:80]
        dest = base / "uploads" / uuid.uuid4().hex[:8]
        dest.mkdir(parents=True, exist_ok=True)
        (dest / name).write_bytes(data)
        return dict(request.query_params), str(dest / name)

    @app.post("/api/ground_video")
    async def ground_video(request: Request) -> dict[str, Any]:
        fn = getattr(service, "ground_demo", None)
        if fn is None:
            raise HTTPException(501, "video input not wired yet")
        body, demo = await _demo_arg(request)
        scenario = known_path(body.get("scenario"), "scenarios")
        try:
            result = await asyncio.to_thread(fn, scenario, demo, body.get("model") or "replay")
        except (ValueError, KeyError) as e:
            raise bad(str(e)) from e
        except (RuntimeError, ImportError, OSError) as e:
            raise bad(f"video grounding failed: {e}") from e
        proposal, rows = result[0], result[1]
        return _flatten(proposal, rows, extraction=_plain(result[2]) if len(result) > 2 else None, demo=demo)

    # -- console: runs -------------------------------------------------------------------------

    def _options_from(req: dict[str, Any], run_id: str, db: str | None, resume: bool = False) -> Any:
        scenario = known_path(req.get("scenario"), "scenarios")
        mode = req.get("mode") or "planner"
        procedure = req.get("procedure")
        if procedure is not None and not isinstance(procedure, dict):
            raise bad("procedure must be an object")
        task = known_path(req["task"], "tasks") if req.get("task") else None
        if procedure is None and not req.get("goal") and not task:
            raise bad("a procedure (accepted plan), goal or task is required")
        kw: dict[str, Any] = {}
        for key in ("interrupt_after", "max_attempts", "max_model_calls"):
            if req.get(key) not in (None, ""):
                try:
                    kw[key] = int(req[key])
                except (TypeError, ValueError) as e:
                    raise bad(f"{key} must be an integer") from e
        mem = req.get("memory") or {}
        if not isinstance(mem, dict):
            raise bad("memory options must be an object")
        return service.RunOptions(
            scenario=scenario, mode=mode, model=req.get("model") or "replay", goal=req.get("goal") or None,
            procedure=procedure, task=task, db=db if mode == "planner" else None, run_id=run_id, resume=resume,
            fail_first=bool(req.get("fail_first")), lie_first=bool(req.get("lie_first")),
            log_dir=str(base / "logs"), memory=mem, **kw)

    def _start(options: Any) -> _Run:
        active = current["run"]
        if active is not None and active.active:
            raise HTTPException(409, f"run {active.run_id} is still active")
        try:
            prepared = service.prepare(options)
        except (ValueError, KeyError) as e:
            raise bad(str(e)) from e
        except (RuntimeError, ImportError, OSError) as e:
            raise bad(f"cannot start run: {e}") from e
        run = _Run(options.run_id, prepared, options.db, options)
        if prepared.procedure is not None:
            run.warnings = list(prepared.procedure.unsupported) + list(prepared.procedure.unresolved)
        ctx_ = prepared.ctx
        ctx_.safety.reset_estop()

        async def _go() -> None:
            run.status = "running"
            try:
                run.report = await service.execute(prepared)
                if run.status == "estopped":
                    return
                run.status = "completed" if getattr(run.report, "completed", False) else (
                    getattr(run.report, "stop_reason", None) or "stopped")
            except asyncio.CancelledError:
                run.status = "estopped"
                raise
            except Exception as e:  # surfaced to the operator, never swallowed
                run.status = "error"
                run.error = f"{type(e).__name__}: {e}"
            finally:
                prepared.ctx.world.log("run_finished", run_id=run.run_id, status=run.status)

        runs[run.run_id] = run
        current["run"] = run
        run.task = asyncio.create_task(_go())
        return run

    @app.post("/api/runs")
    async def start_run(req: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        db = str(base / "console" / f"{run_id}.sqlite3")
        run = _start(_options_from(req, run_id, db))
        return {"run_id": run.run_id, "banner": BANNER, "mode": run.options.mode}

    @app.post("/api/runs/{run_id}/resume")
    async def resume(run_id: str) -> dict[str, Any]:
        old = get_run(run_id)
        if old.active:
            raise HTTPException(409, "run is still active")
        if old.options.mode != "planner" or not old.db:
            raise bad("only planner runs have a ledger to resume")
        opts = dataclasses.replace(old.options, resume=True, interrupt_after=None)
        new = _start(opts)
        return {"run_id": new.run_id, "resumed": True}

    def _store(run: _Run):
        if not run.db or not Path(run.db).exists():
            return None
        from liminal.memory.store import MemoryStore

        return MemoryStore(run.db)

    def _ledger(run: _Run) -> dict[str, Any]:
        out: dict[str, Any] = {"obligations": [], "attempts": [], "frames": []}
        store = _store(run)
        if store is None:
            return out
        try:
            try:
                out["obligations"] = [{"id": o.obligation_id, "status": o.status,
                                       "evidence_ids": list(o.evidence_ids), **_plain(dict(o.payload))}
                                      for o in store.obligations(run.run_id)]
                out["attempts"] = [{"id": a.attempt_id, "status": a.status, "obligation_ids": list(a.obligation_ids),
                                    "evidence_ids": list(a.evidence_ids), **_plain(dict(a.payload))}
                                   for a in store.attempts(run.run_id)]
                out["frames"] = [{"frame_id": f.frame_id, "camera": f.camera, "event": f.event,
                                  "reason": f.selection_reason, "captured_at": f.captured_at,
                                  "metadata": _plain(dict(f.metadata)),
                                  "url": f"/api/frames/{run.run_id}/{f.frame_id}"}
                                 for f in store.selected_frames(run.run_id)]
            except KeyError:  # the run row is created a moment after start
                pass
        finally:
            store.close()
        return out

    @app.get("/api/runs")
    async def list_runs() -> dict[str, Any]:
        return {"runs": [{"run_id": r.run_id, "status": r.status, "mode": r.options.mode, "started": r.started}
                         for r in runs.values()]}

    @app.get("/api/runs/{run_id}")
    async def run_status(run_id: str) -> dict[str, Any]:
        run = get_run(run_id)
        ledger = _ledger(run)
        rep = _plain(run.report) if run.report is not None else None
        if rep and not ledger["obligations"]:
            ledger["obligations"] = [{"id": o["obligation_id"], "status": o["status"],
                                      "evidence_ids": o["evidence_ids"]} for o in rep["obligations"]]
        if rep and not ledger["attempts"]:
            ledger["attempts"] = [{"id": a["attempt_id"], "status": a["status"], "skill": a["skill"],
                                   "arm": a["arm"], "prompt": a["prompt"],
                                   "obligation_ids": a["obligation_ids"]} for a in rep["attempts"]]
        return {"run_id": run.run_id, "status": run.status, "active": run.active, "error": run.error,
                "mode": run.options.mode, "banner": BANNER, "report": rep,
                "summary": run.report.summary() if run.report is not None else None,
                "obligations": ledger["obligations"], "attempts": ledger["attempts"],
                "warnings": run.warnings, "event_count": len(run.events),
                "estopped": run.prepared.ctx.safety.estopped}

    def _events_since(run: _Run, since: int) -> dict[str, Any]:
        evs = run.events
        since = max(0, since)
        return {"events": [{"i": i, **_plain(e)} for i, e in enumerate(evs[since:], start=since)],
                "next": len(evs), "status": run.status, "active": run.active}

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, since: int = 0, stream: int = 0):
        run = get_run(run_id)
        if not stream:
            return _events_since(run, since)

        async def sse():
            n = since
            while True:
                chunk = _events_since(run, n)
                for e in chunk["events"]:
                    yield f"id: {e['i']}\ndata: {json.dumps(e)}\n\n"
                n = chunk["next"]
                if not run.active and n >= len(run.events):
                    yield f"event: done\ndata: {json.dumps({'status': run.status})}\n\n"
                    return
                await asyncio.sleep(0.25)

        return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/runs/{run_id}/scene.png")
    async def scene(run_id: str):
        run = get_run(run_id)
        snap = run.prepared.scene.capture()
        image = snap.images[-1]  # the current view; reference frames come first
        return Response(image.data, media_type=image.media_type, headers={"Cache-Control": "no-store"})

    def _inspect_run():
        try:
            import liminal.memory as memory
        except ImportError:  # pragma: no cover
            return None
        return getattr(memory, "inspect_run", None)

    @app.get("/api/runs/{run_id}/memory")
    async def memory(run_id: str) -> dict[str, Any]:
        run = get_run(run_id)
        out = _ledger(run)
        out["run_id"] = run.run_id
        out["has_ledger"] = bool(run.db)
        inspect = _inspect_run()
        if inspect is not None and run.db and Path(run.db).exists():
            try:
                out["inspect"] = _plain(inspect(run.db, run.run_id))
            except Exception as e:  # inspector is another session's in-progress work
                out["inspect_error"] = f"{type(e).__name__}: {e}"
        return out

    @app.get("/api/frames/{run_id}/{frame_id:path}")
    async def frame(run_id: str, frame_id: str):
        run = get_run(run_id)
        store = _store(run)
        if store is None:
            raise HTTPException(404, "no memory db for this run")
        try:
            f = store.get_frame(run_id, frame_id)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        finally:
            store.close()
        data = f.data
        media = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg" if data[:2] == b"\xff\xd8" else \
            "application/octet-stream"
        return Response(data, media_type=media, headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/demo_frames/{demo}/{stem}")
    async def demo_frame(demo: str, stem: str):
        # Demo keyframe ids look like demo:<name>:<stem>; only shipped demo folders are served.
        if not re.fullmatch(r"[A-Za-z0-9_-]+", demo) or not re.fullmatch(r"[A-Za-z0-9_-]+", stem):
            raise HTTPException(404, "unknown demo frame")
        for ext, media in ((".png", "image/png"), (".jpg", "image/jpeg"), (".jpeg", "image/jpeg")):
            path = Path(service.DEMO_DIR) / demo / "frames" / f"{stem}{ext}"
            if path.is_file():
                return Response(path.read_bytes(), media_type=media, headers={"Cache-Control": "max-age=3600"})
        raise HTTPException(404, "unknown demo frame")

    # -- static console ------------------------------------------------------------------------

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/web/{name}")
    async def web_file(name: str):
        if name not in ("app.js", "style.css", "index.html"):
            raise HTTPException(404, "not found")
        return FileResponse(WEB_DIR / name, headers={"Cache-Control": "no-store"})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return app
