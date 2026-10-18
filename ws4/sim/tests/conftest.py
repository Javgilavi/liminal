import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SIM = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def can_render():
    """Skip rendering tests when no headless GL backend works (set MUJOCO_GL=osmesa on GPU-less machines)."""
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
