"""Local operator UI and task API. Optional web dependencies stay lazy."""

import asyncio
import contextlib
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def create_app(robot: str, backend: str = "mock", policy: str | None = None, task_dir: str = "configs/tasks"):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, Response
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    from liminal.app import build_context
    from liminal.core.types import SO101_JOINTS
    from liminal.hardware.runtime import connected
    from liminal.orchestration import run_task
    from liminal.orchestration.planner import TemplatePlanner

    ctx = build_context(robot, backend=backend, policy=policy)
    planner = TemplatePlanner(task_dir)
    state: dict[str, Any] = {"running": None, "last_report": None, "arms": {}, "sampled_at": 0.0}
    controls = asyncio.Lock()
    stop_sampling = asyncio.Event()

    def sample() -> None:
        arms = {}
        for name, arm in ctx.arms.items():
            try:
                health = arm.health() if hasattr(arm, "health") else {"torque": True, "fault": None}
                arms[name] = {"positions": arm.read_state(), "health": health,
                              "limits": asdict(ctx.safety.limits[name]) if name in ctx.safety.limits else None}
                if health.get("fault") and not ctx.safety.estopped:
                    ctx.safety.estop()
            except Exception as exc:
                arms[name] = {"error": str(exc)}
                if not ctx.safety.estopped:
                    ctx.safety.estop()
        state["arms"], state["sampled_at"] = arms, time.monotonic()

    async def sampler() -> None:
        while not stop_sampling.is_set():
            await asyncio.to_thread(sample)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop_sampling.wait(), 0.1)

    def running() -> bool:
        return state["running"] is not None and not state["running"].done()

    async def cancel_task() -> None:
        task = state["running"]
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    @asynccontextmanager
    async def lifespan(app):
        with connected(ctx):
            await asyncio.to_thread(sample)
            poll = asyncio.create_task(sampler())
            try:
                yield
            finally:
                ctx.safety.estop()
                await cancel_task()
                stop_sampling.set()
                await poll

    app = FastAPI(title="liminal", lifespan=lifespan)
    app.state.context = ctx
    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method != "GET" and origin and urlsplit(origin).netloc != request.headers.get("host"):
            return Response("Cross-origin robot commands are disabled", status_code=403)
        return await call_next(request)

    class RunRequest(BaseModel):
        task: str | None = None
        goal: str | None = None

    class GoalRequest(BaseModel):
        positions: list[float]

    class TorqueRequest(BaseModel):
        enabled: bool

    def manual_arm(name: str):
        if name not in ctx.arms:
            raise HTTPException(404, "unknown arm")
        if running() or ctx.safety.owner_of(name):
            raise HTTPException(409, "arm belongs to a running task")
        return ctx.arms[name]

    def require_ready() -> None:
        if ctx.safety.estopped:
            raise HTTPException(409, "E-stop active; inspect hardware, then reset explicitly")
        if time.monotonic() - state["sampled_at"] > 0.5:
            raise HTTPException(409, "telemetry is stale; command refused")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/tasks")
    def tasks():
        return {"tasks": sorted(p.stem for p in Path(task_dir).glob("*.yaml"))}

    @app.post("/run")
    async def run(req: RunRequest) -> dict[str, Any]:
        async with controls:
            require_ready()
            if running():
                raise HTTPException(409, "a task is already running")
            if any(not getattr(arm, "torque_enabled", True) for arm in ctx.arms.values()):
                raise HTTPException(409, "engage the arms explicitly before running a task")
            try:
                spec = planner.plan(req.task or req.goal or "")
            except (KeyError, ValueError) as exc:
                raise HTTPException(404, str(exc)) from exc

            async def go() -> None:
                try:
                    report = await run_task(spec, ctx)
                    state["last_report"] = asdict(report)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    ctx.safety.estop()
                    state["last_report"] = {"error": str(exc)}

            state["running"] = asyncio.create_task(go())
            return {"started": spec.name}

    @app.post("/arms/{name}/goal")
    async def goal(name: str, req: GoalRequest):
        async with controls:
            require_ready()
            arm = manual_arm(name)
            if not getattr(arm, "torque_enabled", True):
                raise HTTPException(409, "engage the arm explicitly before sending a goal")
            if len(req.positions) != arm.num_joints:
                raise HTTPException(422, "expected six joint values")
            try:
                async with ctx.safety.claim(name, "operator"):
                    await asyncio.to_thread(ctx.safety.send, name, req.positions)
                return {"positions": await asyncio.to_thread(arm.read_state)}
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(409, str(exc)) from exc

    @app.post("/arms/{name}/torque")
    async def torque(name: str, req: TorqueRequest):
        async with controls:
            arm = manual_arm(name)
            if not hasattr(arm, "set_torque"):
                raise HTTPException(409, "torque control is available only on real arms")
            if req.enabled:
                require_ready()
            try:
                await asyncio.to_thread(ctx.safety.set_torque, name, req.enabled)
                return {"torque": arm.torque_enabled}
            except RuntimeError as exc:
                raise HTTPException(409, str(exc)) from exc

    @app.post("/estop")
    async def estop():
        # Latch before waiting for a task or the manual-control lock.
        ctx.safety.estop()
        await cancel_task()
        return {"estopped": True}

    @app.post("/reset")
    async def reset():
        async with controls:
            if running():
                raise HTTPException(409, "wait for the task to stop")
            if any(getattr(a, "fault", None) for a in ctx.arms.values()):
                raise HTTPException(409, "hardware fault remains; inspect hardware before restarting")
            ctx.safety.reset_estop()
            return {"estopped": False}  # reset does not engage motors

    @app.get("/cameras/{name}.jpg")
    def camera(name: str):
        if name not in ctx.cameras:
            raise HTTPException(404, "unknown camera")
        try:
            import cv2
            import numpy as np

            frame = ctx.cameras[name].read()
            frame = np.asarray(frame, dtype=np.uint8)
            ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            return Response(jpeg.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/state")
    async def get_state() -> dict[str, Any]:
        return {
            "facts": ctx.world.snapshot(),
            "owners": {a: ctx.safety.owner_of(a) for a in ctx.arms},
            "estopped": ctx.safety.estopped,
            "running": running(),
            "last_report": state["last_report"],
            "events": ctx.world.events[-50:],
            "arms": state["arms"], "joints": SO101_JOINTS,
            "cameras": list(ctx.cameras),
            "camera_labels": {name: getattr(camera, "label", name) for name, camera in ctx.cameras.items()},
            "sample_age_s": time.monotonic() - state["sampled_at"],
        }

    return app
