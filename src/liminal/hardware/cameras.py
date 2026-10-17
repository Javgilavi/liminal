"""Camera wrappers (owned by WS5). FLUX 3 Action expects a scene and a wrist view per arm."""

from __future__ import annotations

import threading
from typing import Any


class OpenCVCamera:
    def __init__(self, name: str, index: int | str, width: int = 640, height: int = 480, fps: int = 30,
                 rotation: int = 0, label: str | None = None, fourcc: str | None = None):
        if rotation not in (0, 90, 180, 270):
            raise ValueError("camera rotation must be 0, 90, 180 or 270")
        if fourcc is not None and len(fourcc) != 4:
            raise ValueError("camera fourcc must contain four characters")
        self.name = name
        self.label = label or name
        self.rotation = rotation
        self.fourcc = fourcc
        self.index = index
        self.width, self.height, self.fps = width, height, fps
        self._cap: Any = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        import cv2

        backend = cv2.CAP_V4L2 if str(self.index).startswith("/dev/") else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.index, backend)
        if self.fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            raise RuntimeError(f"camera {self.name} ({self.index}) failed to open")

    def read(self) -> Any:
        import cv2

        with self._lock:
            ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"camera {self.name} returned no frame")
        if self.rotation:
            rotation = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
            frame = cv2.rotate(frame, rotation[self.rotation])
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def disconnect(self) -> None:
        if self._cap is not None:
            with self._lock:
                self._cap.release()


class MockCamera:
    def __init__(self, name: str, width: int = 64, height: int = 48):
        self.name = name
        self.width, self.height = width, height

    def read(self) -> Any:
        return [[[0, 0, 0]] * self.width] * self.height
