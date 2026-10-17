"""Live Claude: one planning call and one verifier call on a rendered sim head frame.

Skipped unless ANTHROPIC_API_KEY is set (Oscar exports it himself). Expected cost: a few US cents.
Run: ANTHROPIC_API_KEY=... python -m pytest ws4/orchestrator/tests/test_claude_live.py -q -s
"""

import os

import pytest
from orchestrator.claude import ClaudeJSON
from orchestrator.cost import CostMeter
from orchestrator.plan import Step, concurrent_pairs
from orchestrator.planner import ClaudePlanner
from orchestrator.verifier import ClaudeVerifier

pytestmark = pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")


def test_live_plan_and_verify(vocab, can_render):
    from orchestrator.sim_backend import PackBoxSim

    sim = PackBoxSim(image_size=(480, 640), randomize=False)
    obs = sim.reset(seed=0)
    meter = CostMeter(budget_usd=0.5)
    llm = ClaudeJSON(meter)
    result = ClaudePlanner(vocab, llm).plan("put the apple and the yellow block in the box, then close the lid",
                                            image=obs["observation.images.head"])
    print(f"\nplan: {result.attempts} attempt(s), {result.latency_s:.1f} s, USD {result.cost_usd:.4f}")
    assert result.ok, result.errors
    skills = {s.skill for s in result.plan.steps}
    assert {"put_apple_in_box", "put_block_in_box", "close_lid"} <= skills
    print("steps:", [(s.id, s.arm, s.skill, s.depends_on) for s in result.plan.steps])
    print("concurrent pairs:", concurrent_pairs(result.plan))
    verdict = ClaudeVerifier(llm).check(Step("a", "left", "put_apple_in_box",
                                             success_check="The red apple is inside the box."),
                                        vocab["put_apple_in_box"], obs)
    print("verdict:", verdict.as_dict())
    assert verdict.error is None and not (verdict.done and verdict.confidence >= 0.7)  # the apple is on the table
    print("cost:", meter.summary())
    assert 0 < meter.total_usd < 0.5
