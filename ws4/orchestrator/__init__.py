"""Claude-driven orchestrator for the two XLeRobot arms: plan, approve, execute, verify.

Claude (or the offline stub) turns an operator goal into a plan of per-arm skills from ``skills.yaml``; a human
approves it in the HMI; the executor runs it on both arms through ``flux.dual_arm_runner`` (FLUX skills, or the
scripted sim stand-in), verifies every step, and logs the attempt with ``ws4/eval/trial_log.py``.

Import as ``orchestrator`` with ``ws4/`` on ``sys.path`` (``cd ws4 && python -m orchestrator hmi``). The repo
root is appended too, because ``sim/scripted_pick.py`` imports ``ws4.sim.geometry``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
WS4 = PACKAGE_DIR.parent
REPO = WS4.parent

if str(WS4) not in sys.path:
    sys.path.insert(0, str(WS4))
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))
