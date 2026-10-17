"""Minimal operator page for the fixed five-step pen replacement demo."""
import argparse
import asyncio
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from liminal.maintenance import COLORS, Maintenance
from liminal.settings import load_env_file


def create_app(color='red', robot=None, folder=None):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse

    controller = Maintenance(Path(folder or f'runs/pen-demo/{time.strftime("%Y%m%d-%H%M%S")}'), color, robot)

    @asynccontextmanager
    async def lifespan(app):
        async with AsyncExitStack() as owners:
            if robot:
                await asyncio.to_thread(robot.connect)
                for arm in ('left', 'right'):
                    await owners.enter_async_context(robot.safety.claim(arm, 'pen-maintenance'))
            try:
                yield
            finally:
                controller.stop()
                if app.state.task:
                    await asyncio.gather(app.state.task, return_exceptions=True)
                if robot:
                    await asyncio.to_thread(robot.close)

    app = FastAPI(title='Replace the faulty pen', lifespan=lifespan)
    app.state.task = None
    app.state.controller = controller

    @app.get('/')
    async def index():
        return FileResponse(Path(__file__).parent / 'web/maintenance.html')

    @app.get('/state')
    async def state():
        return controller.state()

    @app.post('/run')
    async def run_step():
        if not robot:
            raise HTTPException(409, 'Photo review only; no robot connected')
        if controller.busy or (app.state.task and not app.state.task.done()):
            raise HTTPException(409, 'A step is already running')
        if controller.status in ('complete', 'stopped'):
            raise HTTPException(409, 'Job is complete or stopped')
        app.state.task = asyncio.create_task(controller.run_job())
        return {'started': controller.index + 1}

    @app.post('/photo')
    async def photo(request: Request):
        # Raw image body avoids another upload dependency; no arbitrary file paths.
        data = await request.body()
        if not data or len(data) > 15_000_000:
            raise HTTPException(422, 'Upload a checkpoint image smaller than 15 MB')
        try:
            await controller.review_photo(data)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return controller.state()

    @app.post('/stop')
    async def stop():
        # Latch immediately; serial holding may wait on a current hardware read.
        controller.cancel.set()
        controller.status = 'stopped'
        await asyncio.to_thread(controller.stop)
        return {'stopped': True}

    @app.get('/checkpoint/{step}')
    async def checkpoint(step: int):
        if not 1 <= step <= 5:
            raise HTTPException(404)
        path = controller.folder / f'checkpoint-{step}.png'
        if not path.exists():
            raise HTTPException(404)
        return FileResponse(path)

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--color', choices=COLORS, default='red')
    parser.add_argument('--port', type=int, default=8780)
    parser.add_argument('--robot', help='robot YAML; omit for photo review with no hardware')
    parser.add_argument('--flux', default='http://127.0.0.1:18100')
    parser.add_argument('--seconds', type=float, default=8, help='bounded duration per policy prompt (1–30)')
    args = parser.parse_args()
    load_env_file()
    robot = None
    if args.robot:
        from liminal.hardware.maintenance import MaintenanceRobot

        robot = MaintenanceRobot(args.robot, args.flux, args.seconds)
    uvicorn.run(create_app(args.color, robot), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
