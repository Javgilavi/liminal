import sys
from pathlib import Path

# ws4/ is not an installed package; make `import kinematics` work from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
