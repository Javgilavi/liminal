"""Small, non-executing loader for local API configuration.

The core package intentionally does not require python-dotenv. Values are never
evaluated as shell code and existing process variables take precedence.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CLAUDE_MODEL = "claude-haiku-5-5"
_ALLOWED = {
    "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "LIMINAL_MAX_MODEL_CALLS",
    "LIMINAL_MAX_OUTPUT_TOKENS", "LIMINAL_MODEL_TIMEOUT_S",
}


def load_env_file(path: str | Path = ".env") -> bool:
    """Read supported KEY=value settings without logging secret values."""
    source = Path(path)
    if not source.is_file():
        return False
    for number, line in enumerate(source.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or key not in _ALLOWED:
            raise ValueError(f"{source}:{number}: expected a supported KEY=value setting")
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"{source}:{number}: unmatched quote")
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        if "\x00" in value:
            raise ValueError(f"{source}:{number}: invalid value")
        os.environ.setdefault(key, value)
    return True


def claude_model() -> str:
    return os.environ.get("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL


def positive_setting(name: str, default: int) -> int:
    value = int(os.environ.get(name) or default)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
