import sys
from pathlib import Path

# ``ws4.imagine`` for these tests, and ws4/ itself so ``flux`` (and ``imagine``, as flux imports it) resolve the way
# the policy server and the runner import them.
REPO_ROOT = Path(__file__).resolve().parents[3]
WS4 = REPO_ROOT / "ws4"
for path in (REPO_ROOT, WS4):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
