"""Arm and Camera protocol views of the Haller arms and cameras owned by the ws4 session."""

from __future__ import annotations

from typing import Any

from liminal.core.types import JointPositions
from liminal.ws4_bridge.session import Ws4Bridge

# ws4 observation keys (flux.contract): the head camera and each robot side's wrist camera
CAMERA_KEYS = {
    "head": "observation.images.head",
    "left_wrist": "observation.images.left_wrist",  # robot LEFT (pen) arm, USB 7.2
    "right_wrist": "observation.images.right_wrist",  # robot RIGHT (box) arm, USB 7.3
}


class Ws4Arm:
    """One Haller arm. State comes from the ws4 session's last observation. It never sends a motor command:
    ``send_action`` raises, so a primitive or another backend cannot bypass ws4 (the supervisor then latches its
    e-stop). Every motion is a ws4 step (``Ws4ChainBackend``) behind the Haller safety stack."""

    num_joints = 6

    def __init__(self, name: str, bridge: Ws4Bridge, ws4_arm: str):
        if ws4_arm not in ("left", "right"):
            raise ValueError(f"{name}: ws4_arm must be the robot side left or right, got {ws4_arm!r}")
        self.name = name
        self.ws4_bridge = bridge
        self.ws4_arm = ws4_arm

    def connect(self) -> None:
        self.ws4_bridge.connect()

    def disconnect(self) -> None:
        self.ws4_bridge.disconnect()

    def read_state(self) -> JointPositions:
        return self.ws4_bridge.joints(self.ws4_arm)

    def send_action(self, positions: JointPositions) -> None:
        raise RuntimeError(f"{self.name}: the Haller arms move only through ws4 steps (backend ws4); "
                           "direct commands are refused")

    def hold(self, positions: JointPositions | None = None) -> None:
        if positions is not None:
            raise RuntimeError(f"{self.name}: holding at a commanded pose would be a direct command; ws4 holds the "
                               "arm where it is")
        self.ws4_bridge.hold(self.ws4_arm)

    def release_hold(self) -> None:
        """Nothing to release: a held arm keeps holding until a ws4 step drives it (the lid closes if it lets go)."""

    def stop(self) -> None:
        self.ws4_bridge.freeze(self.ws4_arm)

    def emergency_stop(self) -> None:
        """Freeze with torque on (never torque off: the box arm would drop the lid)."""
        self.ws4_bridge.estop(f"liminal e-stop ({self.name} arm)")


class Ws4Camera:
    def __init__(self, name: str, bridge: Ws4Bridge, ws4_camera: str):
        if ws4_camera not in CAMERA_KEYS:
            raise ValueError(f"{name}: ws4_camera must be one of {sorted(CAMERA_KEYS)}, got {ws4_camera!r}")
        self.name = name
        self.ws4_bridge = bridge
        self.key = CAMERA_KEYS[ws4_camera]

    def connect(self) -> None:
        self.ws4_bridge.connect()

    def disconnect(self) -> None:
        self.ws4_bridge.disconnect()

    def read(self) -> Any:
        return self.ws4_bridge.image(self.key)
