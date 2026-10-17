"""Camera wrappers (owned by WS5). FLUX 3 Action expects a scene and a wrist view per arm."""

from __future__ import annotations

from typing import Any


class OpenCVCamera:
    def __init__(self, name: str, index: int | str, width: int = 640, height: int = 480, fps: int = 30):
        self.name = name
        self.index = index
        self.width, self.height, self.fps = width, height, fps
        self._cap: Any = None

    def connect(self) -> None:
        import cv2

        self._cap = cv2.VideoCapture(self.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"camera {self.name} ({self.index}) failed to open")

    def read(self) -> Any:
        import cv2

        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"camera {self.name} returned no frame")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()


class MockCamera:
    def __init__(self, name: str, width: int = 64, height: int = 48):
        self.name = name
        self.width, self.height = width, height

    def read(self) -> Any:
        return [[[0, 0, 0]] * self.width] * self.height
