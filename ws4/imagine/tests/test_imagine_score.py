import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PIL")

from ws4.imagine.score import (  # noqa: E402
    DEFAULT_MODEL,
    FALLBACK_BETA,
    ClaudeJudge,
    ColorFractionScorer,
    GoalImageScorer,
    JudgeRefused,
    LatentGoalScorer,
    SurpriseMonitor,
    frame_distance,
    surprise_curve,
)


def square(x0, color=(255, 0, 0)):
    img = np.zeros((64, 128, 3), np.uint8)
    img[20:44, x0 : x0 + 24] = color
    return img


def test_frame_distance_basics():
    a = square(10)
    assert frame_distance(a, a) == 0.0
    assert frame_distance(a, square(80)) > frame_distance(a, square(14))
    assert frame_distance(a, square(80), roi=(0.0, 0.0, 0.05, 0.2)) == 0.0  # empty corner
    curve = surprise_curve([a, a], [a, None])
    assert curve[0] == 0.0 and np.isnan(curve[1])


def test_goal_scorer_prefers_the_candidate_that_reaches_the_goal():
    goal = square(90)
    clips = [np.stack([square(10), square(30)]), np.stack([square(10), square(88)])]
    scores = GoalImageScorer(goal)(clips, current=square(10), offsets=[16, 32])
    assert scores[1] > scores[0] > -1e-6
    absolute = GoalImageScorer(goal, relative=False, offset=16)(clips, offsets=[16, 32])
    assert absolute[0] == pytest.approx(absolute[1])


def test_color_fraction_finds_the_red_cube_in_a_zone():
    scorer = ColorFractionScorer(hue=(240, 15), roi=(0.5, 0.0, 1.0, 1.0))
    assert scorer.predicate(square(90)) and not scorer.predicate(square(10))
    assert scorer([np.stack([square(10), square(90)])]) == [pytest.approx(24 * 24 / (64 * 64))]


def test_surprise_monitor_fires_when_reality_diverges():
    from types import SimpleNamespace

    clip = np.stack([square(10 + 2 * o) for o in range(2, 43)])
    rec = SimpleNamespace(frames=clip, step=100, frame_offsets=tuple(range(2, 43)),
                          frame_at=lambda o: clip[min(max(o - 2, 0), 40)])
    monitor = SurpriseMonitor(threshold=0.02, window=3, min_offset=4)
    assert monitor.update(rec, square(14), 102) == (None, False)  # too soon after the replan
    for step in range(104, 110):  # reality follows the imagination
        value, fired = monitor.update(rec, square(10 + 2 * (step - 100)), step)
        assert value < 0.01 and not fired
    fired_any = False
    for step in range(110, 114):  # the cube stays behind (dropped)
        _, fired = monitor.update(rec, square(10), step)
        fired_any |= fired
    assert fired_any


def test_latent_goal_scorer():
    torch = pytest.importorskip("torch")
    goal = torch.randn(96, 2, 4)
    z_good = torch.randn(96, 3, 2, 4)
    z_good[:, -1] = goal
    scores = LatentGoalScorer(goal)([torch.randn(96, 3, 2, 4), z_good])
    assert scores[1] == pytest.approx(1.0, abs=1e-5) and scores[0] < 0.5


class FakeMessages:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.calls = payload, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = json.dumps(self.payload)
        return SimpleNamespace(stop_reason=self.stop_reason, stop_details=SimpleNamespace(category="cyber"),
                               content=[SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text=text)])


def fake_client(payload, stop_reason="end_turn"):
    messages = FakeMessages(payload, stop_reason)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages), messages=messages), messages


RANKING = {
    "candidates": [
        {"label": "A", "progress": 3, "risk": 2, "plausible": True, "reason": "slow"},
        {"label": "B", "progress": 9, "risk": 2, "plausible": True, "reason": "grasps"},
        {"label": "C", "progress": 10, "risk": 0, "plausible": False, "reason": "cube teleports"},
    ],
    "best_label": "B",
}


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_judge_ranks_candidates_without_network():
    client, messages = fake_client(RANKING)
    judge = ClaudeJudge(client=client, frames_per_candidate=2)
    clips = [np.stack([square(x), square(x + 5), square(x + 10)]) for x in (10, 40, 70)]
    scores = judge(clips, current=square(10), offsets=[2, 22, 42], task="put the cube right")
    assert int(np.argmax(scores)) == 1
    assert scores[2] < scores[1]  # implausible imagination is penalized
    (call,) = messages.calls
    assert call["model"] == DEFAULT_MODEL
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert call["output_config"]["effort"] == "low"
    assert call["output_config"]["format"]["type"] == "json_schema"
    images = [b for b in call["messages"][0]["content"] if b["type"] == "image"]
    assert len(images) == 1 + 3 * 2
    assert any("+1.40 s" in b.get("text", "") for b in call["messages"][0]["content"])


def test_judge_failure_falls_back_to_the_policy_seed():
    client, _ = fake_client(RANKING, stop_reason="refusal")
    judge = ClaudeJudge(client=client)
    clips = [np.stack([square(10)])] * 3
    assert judge(clips, current=None, offsets=[42], task="x") == [1.0, 0.0, 0.0]
    assert "JudgeRefused" in judge.last_error
    with pytest.raises(JudgeRefused):
        judge.rank(clips, offsets=[42], task="x")


def test_anticipate_and_compare_parse_json():
    client, messages = fake_client({"success_probability": 0.8, "imagined_outcome": "cube in bowl",
                                    "failure_mode": "none", "plausible": True})
    judge = ClaudeJudge(client=client, use_fallbacks=False)
    out = judge.anticipate(np.stack([square(10), square(90)]), current=square(10), offsets=[2, 42], task="t")
    assert out["success_probability"] == 0.8 and "betas" not in messages.calls[0]
    client, messages = fake_client({"matches_imagination": False, "task_done": False, "difference": "dropped",
                                    "recommendation": "retry"})
    assert ClaudeJudge(client=client).compare(square(90), square(10), task="t")["recommendation"] == "retry"


def test_judge_construction_is_lazy():
    judge = ClaudeJudge()
    assert judge._client is None  # no SDK import, no key needed until the first call


def test_speed_check_summary_ratios():
    from ws4.imagine.speed_check import summarize

    rows = [{"pred_path": [2, 1, 1, 1, 1, 0], "rec_path": [1, 2, 2, 1, 1, 0],
             "pred_disp": [2, 1, 1, 1, 1, 0], "rec_disp": [1, 2, 2, 1, 1, 0]}] * 3  # fmt: skip
    s = summarize(rows)
    assert s["moving_chunks"] == 3 and s["ratio_path_model_over_demo"][:3] == [2.0, 0.5, 0.5]
    assert s["median_arm_travel_deg"] == {"model": 6.0, "demo": 7.0} and s["fraction_model_travels_more"] == 0.0
