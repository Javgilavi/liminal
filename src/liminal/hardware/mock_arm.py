"""In-memory arm for tests, CI and orchestration development without hardware."""

from __future__ import annotations

from liminal.core.types import SO101_JOINTS, JointPositions


class MockArm:
    def __init__(self, name: str, num_joints: int = 6, home: JointPositions | None = None):
        self.name = name
        self.num_joints = num_joints
        self.policy_contract = {"embodiment": "so101", "state_units": "degrees_percent",
                                "joint_order": SO101_JOINTS if num_joints == 6 else
                                tuple(f"joint_{i}" for i in range(num_joints))}
        self.home = list(home) if home else [0.0] * num_joints
        self.state: JointPositions = list(self.home)
        self.connected = False
        self.holding: JointPositions | None = None
        self.stopped = False
        self.history: list[JointPositions] = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_state(self) -> JointPositions:
        return list(self.state)

    def send_action(self, positions: JointPositions) -> None:
        if len(positions) != self.num_joints:
            raise ValueError(f"{self.name}: expected {self.num_joints} joints, got {len(positions)}")
        self.state = list(positions)
        self.history.append(list(positions))

    def hold(self, positions: JointPositions | None = None) -> None:
        self.holding = list(positions) if positions else self.read_state()
        self.state = list(self.holding)

    def release_hold(self) -> None:
        self.holding = None

    def stop(self) -> None:
        self.stopped = True
        self.holding = self.read_state()
