"""Start the dependency/model setup remotely, without saving a Hugging Face token.

Run after copying runpod_setup.sh, download_models.py and the pinned LeRobot source
as described in RUNPOD_FLUX.md. The setup process detaches and logs on the Pod.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    token = os.environ.get("HF_TOKEN")
    if not token:
        source = Path(os.environ.get("HF_TOKEN_PATH", str(Path.home() / ".cache/huggingface/token")))
        token = source.read_text().strip() if source.is_file() else None
    if not token:
        raise SystemExit("A Hugging Face login with access to both FLUX repositories is required.")
    remote = """
import os, sys, subprocess
from pathlib import Path
root=Path('/workspace/flux3-action')
pid_file=root/'setup.pid'
if pid_file.exists():
    stat=Path('/proc')/pid_file.read_text().strip()/'stat'
    if stat.exists() and stat.read_text().split()[2]!='Z':
        raise SystemExit('Setup already running; inspect /workspace/flux3-action/setup.log')
token=sys.stdin.read().strip()
assert token, 'Hugging Face credential unavailable'
with (root/'setup.log').open('ab',buffering=0) as log:
    log.write(b'\\n--- New FLUX setup attempt ---\\n')
    proc=subprocess.Popen(['bash',str(root/'runpod_setup.sh')], stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env={**os.environ,'HF_TOKEN':token})
pid_file.write_text(str(proc.pid))
print('Remote setup started; PID',proc.pid)
"""
    command = [
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={here / '.runpod/known_hosts'}", "-o", "ConnectTimeout=15",
        "-i", os.environ.get("RUNPOD_SSH_KEY", str(Path.home() / ".ssh/id_ed25519")),
        "-p", os.environ.get("RUNPOD_SSH_PORT", "11752"),
        "root@" + os.environ.get("RUNPOD_SSH_HOST", "154.54.102.27"),
        "python3 -c " + shlex.quote(remote),
    ]
    result = subprocess.run(command, input=token.encode(), capture_output=True, timeout=30, check=False)
    print(result.stdout.decode(), end="")
    print(result.stderr.decode(), end="")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
