"""Operator HTTP API (owned by WS5). A web page, voice front-end, or a judge's phone
can all trigger tasks through the same three calls:

    POST /run   {"task": "lid_hold_insert"}  or  {"goal": "put the block in the box"}
    POST /estop
    GET  /state  -> world facts, arm owners, last report, recent events
"""

from __future__ import annotations

import asyncio
from typing import Any


def create_app(robot: str, backend: str = "mock", policy: str | None = None, task_dir: str = "configs/tasks"):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    from liminal.app import build_context
    from liminal.orchestration import run_task
    from liminal.orchestration.planner import TemplatePlanner

    app = FastAPI(title="liminal")
    ctx = build_context(robot, backend=backend, policy=policy)
    planner = TemplatePlanner(task_dir)
    state: dict[str, Any] = {"running": None, "last_report": None}

    class RunRequest(BaseModel):
        task: str | None = None
        goal: str | None = None

    @app.on_event("startup")
    def _connect() -> None:
        for a in ctx.arms.values():
            a.connect()

    @app.post("/run")
    async def run(req: RunRequest) -> dict[str, Any]:
        if state["running"] is not None and not state["running"].done():
            raise HTTPException(409, "a task is already running")
        try:
            spec = planner.plan(req.task or req.goal or "")
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
        return {"estopped": True}

    @app.get("/state")
    async def get_state() -> dict[str, Any]:
        return {
            "facts": ctx.world.snapshot(),
            "owners": {a: ctx.safety.owner_of(a) for a in ctx.arms},
            "estopped": ctx.safety.estopped,
            "running": state["running"] is not None and not state["running"].done(),
            "last_report": state["last_report"],
            "events": ctx.world.events[-50:],
        }

    return app
