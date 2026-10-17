import sys
from pathlib import Path

# Make ``ws4.imagine`` importable when pytest is run from anywhere (ws4 is a namespace package).
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
