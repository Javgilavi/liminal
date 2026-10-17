import pytest

from liminal.app import build_context
from liminal.policy.mock import MockBackend


@pytest.fixture
def mock_backend():
    return MockBackend(duration_s=0.01)


@pytest.fixture
def make_ctx(mock_backend):
    def _make(backend=None, robot="configs/robot/mock_dual.yaml"):
        return build_context(robot, policy_backend=backend or mock_backend)

    return _make
