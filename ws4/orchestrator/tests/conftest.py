"""Put ws4/ on sys.path (``import orchestrator``, ``flux``, ``sim``, ``kinematics``) and share small fakes."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
WS4 = Path(__file__).resolve().parents[2]
if str(WS4) not in sys.path:
    sys.path.insert(0, str(WS4))

import orchestrator  # noqa: E402,F401  (path setup)
from flux.contract import ARM_DIM, IMG_HEAD, IMG_LEFT_WRIST, IMG_RIGHT_WRIST, OBS_STATE  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls the real Claude API (needs a key; costs money)")


@pytest.fixture(autouse=True)
def no_api_key_unless_live(request, monkeypatch, tmp_path):
    """Offline tests must never reach the API: hide the env key and point the key file at nothing."""
    if request.node.get_closest_marker("live") is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", str(tmp_path / "no-key.env"))


@pytest.fixture(scope="session")
def can_render():
    """Skip rendering tests when no headless GL backend works (MUJOCO_GL=osmesa on GPU-less machines)."""
    import mujoco

    try:
        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='0.1'/></worldbody></mujoco>")
        r = mujoco.Renderer(m, 16, 16)
        r.update_scene(mujoco.MjData(m))
        r.render()
        r.close()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no headless OpenGL ({os.environ.get('MUJOCO_GL')}): {e}")
    return True


@pytest.fixture(scope="session")
def vocab():
    from orchestrator.skills import SIM_VOCABULARY, load_vocabulary

    return load_vocabulary(SIM_VOCABULARY)  # the sim pack-box vocabulary; test_haller.py covers skills.yaml


class FollowRobot:
    """Robot whose state is exactly the last command; tiny static images. Counts commands sent."""

    def __init__(self, state12=None, hw=(24, 32)):
        self.state = np.zeros(2 * ARM_DIM) if state12 is None else np.asarray(state12, float).copy()
        img = np.full((*hw, 3), 100, np.uint8)
        self.images = {IMG_HEAD: img, IMG_LEFT_WRIST: img, IMG_RIGHT_WRIST: img}
        self.sent = 0

    def get_observation(self):
        return {OBS_STATE: self.state.copy(), **self.images}

    def send_action(self, action):
        self.state = np.asarray(action, float).copy()
        self.sent += 1


@pytest.fixture
def follow_robot():
    return FollowRobot


class FakeClaude:
    """Stands in for ``anthropic.Anthropic``: returns queued JSON payloads, records every request."""

    def __init__(self, payloads, usage=None, stop_reason="end_turn", model=None):
        self.payloads = list(payloads)
        self.requests = []
        self.usage = usage or {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 0}
        self.stop_reason = stop_reason
        self.model = model
        self.messages = SimpleNamespace(create=lambda **kw: self._create(False, kw))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: self._create(True, kw)))

    def _create(self, beta, kwargs):
        self.requests.append({"beta": beta, **kwargs})
        payload = self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(**self.usage), stop_reason=self.stop_reason,
            stop_details=SimpleNamespace(category="bio") if self.stop_reason == "refusal" else None,
            model=self.model or kwargs["model"],
        )


@pytest.fixture
def fake_claude():
    return FakeClaude
