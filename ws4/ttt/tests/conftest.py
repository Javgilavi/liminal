import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

# ``ws4/`` on the path so the tests import the prototype as the ``ttt`` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
