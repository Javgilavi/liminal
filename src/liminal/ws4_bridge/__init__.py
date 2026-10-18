"""Bridge from the liminal behavior tree to the ws4 Haller stack (ws4/orchestrator, ws4/single_arm, ws4/flux).

The tree decides what runs, when and on which arm; ws4 executes every motion. One ws4 orchestrator ``Session`` owns
both Haller arms and the cameras for the whole run (``session.Ws4Bridge``). A skill node becomes a one-step ws4 plan
(``backend.Ws4ChainBackend``), executed by the ws4 executor behind the full Haller safety stack: per-skill keep-out
envelopes, the other arm as a live capsule, the box-arm start gate, stall / load / keep-out freezes, the torque cap,
the cutter slow zone, the lid precheck and the held box arm. FLUX runs on the remote policy server (ws4/cloud,
port 8790 through an SSH tunnel), one LoRA package per skill. An e-stop freezes both arms with torque on (grippers
keep their grip, so the lid stays held); nothing here ever switches torque off.

Frame: liminal task files and ws4 both use the ROBOT's sides. ``left`` = the pen arm = /dev/haller_arm_right, wrist
camera USB 7.2; ``right`` = the box arm = /dev/haller_arm_left, wrist camera USB 7.3. The recorder and operator frame
(``left_*`` = robot right) never appears here.

ws4 is imported lazily (heavy deps, hardware); ``LIMINAL_WS4_DIR`` overrides where it lives.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def ws4_dir() -> Path:
    return Path(os.environ.get("LIMINAL_WS4_DIR") or REPO / "ws4").resolve()


def import_orchestrator():
    """Put ws4/ on sys.path (as ``cd ws4 && python -m orchestrator`` does) and return the ``orchestrator`` package."""
    root = ws4_dir()
    if not (root / "orchestrator" / "__init__.py").is_file():
        raise RuntimeError(f"ws4 orchestrator not found under {root} (set LIMINAL_WS4_DIR)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import orchestrator  # noqa: PLC0415

    return orchestrator
