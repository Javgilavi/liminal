"""Live Claude: one minimal call per feature (planner structured output on a camera frame, vision checks).

Runs only when an API key is available (``ANTHROPIC_API_KEY`` or ``~/.config/ws4/anthropic.env``, see claude.py),
because it costs money (about USD 0.10 to 0.30). The measurements land in ``runs/live_claude_<time>.json``.
Run: ``cd ws4 && ORCH_LIVE=1 python -m pytest orchestrator/tests/test_claude_live.py -q -s``
"""

import os

import pytest
from orchestrator.claude import api_key_available

# Opt-in: an exported ANTHROPIC_API_KEY (as in the brief) or ORCH_LIVE=1 with the key file, never by accident.
ENABLED = bool(os.environ.get("ANTHROPIC_API_KEY")) or os.environ.get("ORCH_LIVE") == "1"
pytestmark = [pytest.mark.live,
              pytest.mark.skipif(not (ENABLED and api_key_available()),
                                 reason="live test: export ANTHROPIC_API_KEY or set ORCH_LIVE=1 (costs money)")]


def test_live_plan_and_checks(can_render):
    from orchestrator.live_check import run

    report = run(budget_usd=0.5)
    calls = {c["call"]: c for c in report["calls"]}
    for c in report["calls"]:
        print(c["call"], c["model"], c["latency_s"], "s", c["cost_usd"], "USD", "got", c["got"])
    assert calls["plan (cold cache)"]["got"] == "valid"
    assert calls["plan (warm cache)"]["cache_read_tokens"] > 0  # the system prompt + vocabulary is cached
    for name, c in calls.items():
        if name.startswith("verify"):
            assert c["error"] is None, c
    assert 0 < report["summary"]["total_usd"] < 0.5
