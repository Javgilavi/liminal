"""Put ws4/ on sys.path so the package imports as ``flux`` (and ws4/kinematics as ``kinematics``)."""

import sys
from pathlib import Path

WS4 = Path(__file__).resolve().parents[2]
if str(WS4) not in sys.path:
    sys.path.insert(0, str(WS4))
