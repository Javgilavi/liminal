"""Claude planner and verifier against a fake client: feedback loop, request shape, cost, budget, refusals."""

import numpy as np
import pytest
from orchestrator.claude import FALLBACK_BETA, ClaudeJSON, ClaudeRefused
from orchestrator.cost import PRICES, BudgetExceeded, CostMeter, cost_of
from orchestrator.plan import Step
from orchestrator.planner import ClaudePlanner
from orchestrator.verifier import ClaudeVerifier

GOAL = "put the apple and the yellow block in the box, then close the lid"


def step(id_, arm, skill, deps=(), check="x"):
    return {"id": id_, "arm": arm, "skill": skill, "depends_on": list(deps), "hold_until": [],
            "success_check": check, "max_attempts": 2}


VALID = {"rationale": "both arms pick in parallel, then the right arm closes",
         "steps": [step("apple", "left", "put_apple_in_box"), step("block", "right", "put_block_in_box"),
                   step("lid", "right", "close_lid", ["apple", "block"])]}
INVALID = {"rationale": "", "steps": [step("apple", "right", "open_lid"), step("block", "right", "put_block_in_box")]}


def test_planner_feeds_validator_errors_back(vocab, fake_claude):
    client = fake_claude([INVALID, VALID])
    planner = ClaudePlanner(vocab, ClaudeJSON(CostMeter(1.0), client=client))
    image = np.zeros((48, 64, 3), np.uint8)
    result = planner.plan(GOAL, image=image)
    assert result.ok and result.attempts == 2
    first, second = client.requests
    # Opus 5.5: beta endpoint with server-side refusal fallbacks, adaptive thinking, structured output, cached system.
    assert first["beta"] and first["betas"] == [FALLBACK_BETA] and first["fallbacks"] == "default"
    assert first["model"] == "claude-opus-5-5" and first["thinking"] == {"type": "adaptive"}
    assert first["output_config"]["format"]["type"] == "json_schema"
    assert first["output_config"]["effort"] == "medium"
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert any(b.get("type") == "image" for b in first["messages"][0]["content"])
    # Second call appends the assistant turn and the validator's errors (append-only history).
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
    assert "can only run on" in second["messages"][2]["content"]
    assert result.cost_usd == pytest.approx(2 * cost_of("claude-opus-5-5", client.usage))


def test_planner_gives_up_after_two_feedback_rounds(vocab, fake_claude):
    client = fake_claude([INVALID])
    result = ClaudePlanner(vocab, ClaudeJSON(CostMeter(1.0), client=client)).plan(GOAL)
    assert not result.ok and result.attempts == 3 and len(client.requests) == 3


def test_budget_guard_blocks_calls(vocab, fake_claude):
    client = fake_claude([VALID])
    meter = CostMeter(budget_usd=0.0)
    with pytest.raises(BudgetExceeded):
        ClaudePlanner(vocab, ClaudeJSON(meter, client=client)).plan(GOAL)
    assert client.requests == []


def test_cost_table_and_meter():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000,
             "cache_creation_input_tokens": 1_000_000}
    assert cost_of("claude-haiku-5-5", usage) == pytest.approx(sum(PRICES["claude-haiku-5-5"]))
    meter = CostMeter(1.0)
    meter.record("verify", "claude-haiku-5-5", {"input_tokens": 500, "output_tokens": 50}, 0.8)
    summary = meter.summary()
    assert summary["by_purpose"]["verify"]["calls"] == 1
    assert summary["total_usd"] == pytest.approx((500 * 0.10 + 50 * 0.50) / 1e6, abs=1e-6)


def test_refusal_raises(fake_claude):
    llm = ClaudeJSON(CostMeter(1.0), client=fake_claude([VALID], stop_reason="refusal"))
    with pytest.raises(ClaudeRefused):
        llm.call(purpose="plan", model="claude-opus-5-5", system="s", messages=[{"role": "user", "content": "x"}],
                 schema={"type": "object"})


def test_verifier_uses_the_skill_model_and_never_raises(vocab, fake_claude):
    obs = {"observation.images.head": np.zeros((48, 64, 3), np.uint8)}
    s = Step("apple", "left", "put_apple_in_box", success_check="The red apple is inside the box.")
    client = fake_claude([{"done": True, "confidence": 0.93, "observation": "apple in box"}])
    verdict = ClaudeVerifier(ClaudeJSON(CostMeter(1.0), client=client)).check(s, vocab["put_apple_in_box"], obs)
    assert verdict.done and verdict.confidence == pytest.approx(0.93) and verdict.source == "claude-haiku-5-5"
    req = client.requests[0]
    assert not req["beta"]  # Haiku 5.5 has no server-side fallback
    assert req["output_config"]["effort"] == "low"
    assert "The red apple is inside the box." in req["messages"][0]["content"][-1]["text"]
    # Close-lid checks use Sonnet 5.5 (beta path with fallbacks).
    lid = Step("lid", "right", "close_lid", success_check="lid on")
    ClaudeVerifier(ClaudeJSON(CostMeter(1.0), client=client)).check(lid, vocab["close_lid"], obs)
    assert client.requests[1]["model"] == "claude-sonnet-5-5" and client.requests[1]["beta"]
    refusing = fake_claude([{"done": True}], stop_reason="refusal")
    v = ClaudeVerifier(ClaudeJSON(CostMeter(1.0), client=refusing)).check(s, vocab["put_apple_in_box"], obs)
    assert not v.done and "ClaudeRefused" in v.error
    broke = ClaudeVerifier(ClaudeJSON(CostMeter(0.0), client=client)).check(s, vocab["put_apple_in_box"], obs)
    assert not broke.done and "BudgetExceeded" in broke.error


def test_missing_key_is_reported_not_crashing(vocab):
    obs = {"observation.images.head": np.zeros((8, 8, 3), np.uint8)}
    v = ClaudeVerifier(ClaudeJSON(CostMeter(1.0))).check(Step("a", "left", "put_apple_in_box", success_check="x"),
                                                         vocab["put_apple_in_box"], obs)
    assert not v.done and "ClaudeUnavailable" in v.error


def test_shadow_verifier_logs_agreement(vocab, fake_claude):
    import time

    from orchestrator.verifier import ShadowVerifier

    class Truth:
        name, fast = "truth", True

        def __init__(self, done):
            self.done = done

        def check(self, step, skill, obs):
            from orchestrator.verifier import Verdict

            return Verdict(self.done, 1.0, "truth", "truth")

    obs = {"observation.images.head": np.zeros((32, 32, 3), np.uint8)}
    s = Step("a", "left", "put_apple_in_box", success_check="x")
    reports = []
    claude = ClaudeVerifier(ClaudeJSON(CostMeter(1.0), client=fake_claude(
        [{"done": True, "confidence": 0.9, "observation": "in"}])))
    for truth in (True, False):
        shadow = ShadowVerifier(Truth(truth), claude, lambda st, p, o: reports.append((p.done, o.done)))
        assert shadow.check(s, vocab["put_apple_in_box"], obs).done is truth  # the primary decides
        deadline = time.time() + 5
        while len(shadow.pairs) < 1 and time.time() < deadline:
            time.sleep(0.01)
        agreement = shadow.agreement()
        assert agreement["checks"] == 1
        assert agreement["false_positive"] == (0 if truth else 1)
    assert reports == [(True, True), (False, True)]


def test_key_resolution_env_then_file(monkeypatch, tmp_path):
    from orchestrator.claude import ClaudeUnavailable, api_key_available, resolve_api_key

    assert not api_key_available()  # the autouse fixture hides any real key
    with pytest.raises(ClaudeUnavailable) as err:
        resolve_api_key()
    assert "ANTHROPIC_API_KEY" in str(err.value)
    key_file = tmp_path / "anthropic.env"
    key_file.write_text("# comment\nexport ANTHROPIC_API_KEY='sk-test-file'\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", str(key_file))
    assert resolve_api_key() == "sk-test-file"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-env")
    assert resolve_api_key() == "sk-test-env"  # the environment wins


def test_image_tokens_follow_the_patch_formula():
    from orchestrator.claude import image_tokens

    assert image_tokens(640, 480) == 23 * 18 == 414
    assert image_tokens(1000, 1000) == 1296  # vision docs table
    assert image_tokens(200, 200) == 64


def test_verify_eval_measures_exactly_what_the_live_verifier_sends():
    """verify_eval's tuned variant and ClaudeVerifier build the same request from skills.yaml (no drift)."""
    from orchestrator.plan import Step
    from orchestrator.skills import DEFAULT_VOCABULARY, load_vocabulary
    from orchestrator.verifier import HEAD, ClaudeVerifier
    from orchestrator.verify_eval import CHECKS, VARIANTS, Sample

    vocab = load_vocabulary(DEFAULT_VOCABULARY)
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    live = ClaudeVerifier(llm=None, station=vocab.station)
    tuned = VARIANTS["live"]
    assert live.system == tuned.system == vocab.station.verify_scene
    for check, skill_name in (("put", "put_pen_in_slot"), ("take", "take_pen_out")):
        skill = vocab[skill_name]
        assert skill.success_check == CHECKS[check]
        step = Step("s", "left", skill_name, success_check=skill.success_check)
        sample = Sample("x", frame, None, {check: True})
        assert live.content(step, {HEAD: frame}, skill) == tuned.content(check, sample)
