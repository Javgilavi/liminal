"""Control-contract checks only; these doubles do not establish manipulation success."""
import asyncio
import io
import threading

import pytest

from liminal.hardware.maintenance import MaintenanceRobot
from liminal.maintenance import Maintenance, steps


def png():
    from PIL import Image

    buffer = io.BytesIO()
    Image.new('RGB', (16, 16)).save(buffer, format='PNG')
    return buffer.getvalue()


class Verifier:
    def __init__(self, value=True):
        self.value = value

    def check(self, step, color, current, reference):
        return {'checks': dict.fromkeys(step.checks, self.value), 'reason': 'Unit-test interface double'}


class Robot:
    """Records the driver contract; it does not simulate a scene or evaluate images."""
    left_hold = False

    def __init__(self):
        self.calls = []

    def perform(self, index, step, cancel):
        if index in (2, 3):
            assert self.left_hold, 'left hold must survive both worker steps'
        self.calls += [(step.arm, prompt) for prompt in step.prompts]
        if index == 1:
            self.left_hold = True
        if index == 4:
            self.left_hold = False

    def photo(self):
        return png()

    def stop(self):
        pass


def test_full_sequence_routes_six_prompts_with_hold(tmp_path):
    robot = Robot()
    job = Maintenance(tmp_path, 'blue', robot, Verifier())
    asyncio.run(job.run_job())
    assert job.status == 'complete'
    assert len(job.world.completed) == 5
    assert [arm for arm, _ in robot.calls] == ['left', 'left', 'right', 'right', 'left', 'left']
    assert robot.calls[2][1] == 'Pick up the blue pen and put it in the faulty cup'
    assert len(list(tmp_path.glob('checkpoint-*.png'))) == 5
    assert robot.calls[-2:]== [('left', 'Release the cable'), ('left', 'Close the box lid')]


@pytest.mark.parametrize('verdict', [False, None, 'true'])
def test_unverified_photo_never_advances(tmp_path, verdict):
    job = Maintenance(tmp_path, verifier=Verifier(verdict))
    asyncio.run(job.review_photo(png()))
    assert job.status == 'paused'
    assert job.index == 0
    assert job.world.completed == []


def test_late_visual_confirmation_cannot_advance_after_stop(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class SlowVerifier(Verifier):
        def check(self, *args):
            entered.set()
            assert release.wait(3)
            return super().check(*args)

    job = Maintenance(tmp_path, verifier=SlowVerifier())

    async def exercise():
        task = asyncio.create_task(job.review_photo(png()))
        await asyncio.to_thread(entered.wait, 2)
        job.stop()
        release.set()
        await task

    asyncio.run(exercise())
    assert job.status == 'stopped'
    assert job.world.completed == []


def test_real_driver_refuses_worker_without_scripted_hold():
    robot = MaintenanceRobot.__new__(MaintenanceRobot)
    robot.stopped, robot.left_hold = threading.Event(), False
    with pytest.raises(RuntimeError, match='left scripted hold'):
        robot.perform(2, steps('red')[2], threading.Event())


def test_photo_review_api_uses_body_and_persists_latest_checkpoint(tmp_path):
    from fastapi.testclient import TestClient

    from liminal.ui.maintenance import create_app

    app = create_app(folder=tmp_path)
    app.state.controller.verifier = Verifier()
    with TestClient(app) as client:
        assert client.post('/run').status_code == 409
        response = client.post('/photo', content=png(), headers={'Content-Type': 'image/png'})
        assert response.status_code == 200
        assert response.json()['step'] == 2
        assert response.json()['checkpoint'] == 1
        assert client.get('/checkpoint/1').status_code == 200
