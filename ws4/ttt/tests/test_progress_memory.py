import torch
from ttt.progress_memory import EpisodeProgressMemory, fit, synthetic_aliased_episodes


def test_shapes_and_streaming_equals_sequence():
    torch.manual_seed(0)
    model = EpisodeProgressMemory(16, 6, dim=32).eval()
    feats = torch.randn(2, 7, 16)
    out, _ = model(feats)
    assert out["subtask"].shape == (2, 7, 6) and out["done"].shape == (2, 7)
    state, steps = None, []
    for t in range(7):  # one call per FLUX re-plan
        o, state = model.step(feats[:, t], state)
        steps.append(o["subtask"])
    torch.testing.assert_close(torch.stack(steps, 1), out["subtask"])


def test_synthetic_episodes_alias_repeated_stages():
    data = synthetic_aliased_episodes(4, feat_dim=8, objects=2, noise=0.0, seed=0)
    f, y = data["feats"][0], data["subtask"][0]
    first = f[y == 1][0]
    second = f[y == 5][0]  # the same stage type for the second object
    torch.testing.assert_close(first, second)
    assert int(data["done"][0][y >= 0].sum()) == 8


def test_fit_runs_and_lowers_the_loss():
    torch.manual_seed(0)
    data = synthetic_aliased_episodes(32, feat_dim=8, objects=2, seed=0)
    model = EpisodeProgressMemory(8, 8, dim=32)
    losses = fit(model, data["feats"], data["subtask"], data["done"], steps=30, batch=8, segment=16)
    assert sum(losses[-5:]) < sum(losses[:5])
