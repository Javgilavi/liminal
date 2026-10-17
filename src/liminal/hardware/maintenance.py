"""Thin real-arm driver: shared FLUX, idle-arm hold, and one scene checkpoint."""
from __future__ import annotations

import base64
import io
import math
import threading
import time
from pathlib import Path

import yaml

from liminal.hardware.cameras import OpenCVCamera
from liminal.hardware.so101 import SO101Arm
from liminal.safety.supervisor import JointLimits, SafetySupervisor


class MaintenanceRobot:
    def __init__(self, config: str, policy_url: str, seconds: float):
        if not 1 <= seconds <= 30:
            raise ValueError('Each policy prompt must run for 1–30 seconds')
        cfg = yaml.safe_load(Path(config).read_text())
        self.arms = {name: SO101Arm(name, item['port'], item['id'], max_relative_target=4, use_degrees=True)
                     for name, item in cfg['arms'].items()}
        if set(self.arms) != {'left', 'right'}:
            raise ValueError('Exactly two arms are required: left and right')
        self.cameras = {name: OpenCVCamera(name, **item) for name, item in cfg['cameras'].items()}
        if set(self.cameras) != {'top', 'left_wrist', 'right_wrist'}:
            raise ValueError('Need top, left_wrist and right_wrist cameras')
        limits = {name: JointLimits(**cfg['limits'][name]) for name in self.arms}
        self.safety = SafetySupervisor(self.arms, limits)
        self.url, self.seconds = policy_url.rstrip('/'), seconds
        self.left_hold = False
        self.stopped = threading.Event()
        self._command_lock = threading.RLock()

    def connect(self):
        import httpx

        response = httpx.get(self.url + '/health', timeout=10)
        response.raise_for_status()
        info = response.json()
        if info.get('fake') is not False or 'flux' not in str(info.get('model', '')).lower():
            raise ValueError('Expected the real FLUX SO-101 HTTP server')
        try:
            for camera in self.cameras.values():
                camera.connect()
            for arm in self.arms.values():
                arm.connect()
                arm.hold()
        except BaseException:
            self.close()
            raise

    def picture(self, role: str, format: str = 'JPEG') -> bytes:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(self.cameras[role].read()).save(buffer, format=format)
        return buffer.getvalue()

    def photo(self) -> bytes:
        return self.picture('top', 'PNG')

    def perform(self, index, step, cancel):
        import httpx

        if self.stopped.is_set():
            raise RuntimeError('Stopped: reset the physical scene before starting another job')
        if index in (2, 3) and (not self.left_hold or not self.arms['left'].holding):
            raise RuntimeError('Worker cannot run without the left scripted hold')
        arm = self.arms[step.arm]
        # The other arm remains in its local hold thread; it receives no policy prompts.
        with httpx.Client(timeout=20) as client:
            try:
                for prompt in step.prompts:
                    if cancel.is_set() or self.stopped.is_set():
                        return
                    response = client.post(self.url + '/reset', json={'task': prompt})
                    response.raise_for_status()
                    # Keep holding while the remote model encodes the new instruction.
                    with self._command_lock:
                        if cancel.is_set() or self.stopped.is_set():
                            return
                        arm.release_hold()
                        if step.arm == 'left':
                            self.left_hold = False
                    deadline = time.monotonic() + self.seconds
                    while time.monotonic() < deadline and not cancel.is_set() and not self.stopped.is_set():
                        response = client.post(self.url + '/act', json={
                            'task': prompt, 'state': arm.read_state(),
                            'scene': base64.b64encode(self.picture('top')).decode(),
                            'wrist': base64.b64encode(self.picture(f'{step.arm}_wrist')).decode()})
                        response.raise_for_status()
                        action = response.json().get('action')
                        if (not isinstance(action, list) or len(action) != arm.num_joints
                                or any(type(x) not in (int, float) or not math.isfinite(x) for x in action)):
                            raise ValueError('FLUX must return six finite joint targets')
                        with self._command_lock:
                            if cancel.is_set() or self.stopped.is_set() or time.monotonic() >= deadline:
                                break
                            self.safety.send(step.arm, action)
                        time.sleep(1 / 30)
                    arm.hold()
            finally:
                arm.hold()
                if step.arm == 'left':
                    # On step 2, freeze exactly where the cable move finished.
                    self.left_hold = index == 1 or (index == 4 and cancel.is_set())

    def stop(self):
        # Set the latch before waiting on serial I/O; the policy loop rechecks it under the same lock.
        self.stopped.set()
        with self._command_lock:
            self.safety.estop()

    def close(self):
        for arm in self.arms.values():
            if arm._robot is not None:
                arm.disconnect()
        for camera in self.cameras.values():
            camera.disconnect()
