import numpy as np
import pytest

pytest.importorskip("PIL")
torch = pytest.importorskip("torch")

from ws4.imagine.compose import (  # noqa: E402
    ActualFrame,
    Renderer,
    align,
    contact_sheet,
    make_canvas,
    render_run,
    write_gif,
)
from ws4.imagine.fakes import FakeFluxPolicy, fake_batch  # noqa: E402
from ws4.imagine.predict_with_video import Imaginer  # noqa: E402


@pytest.fixture
def records():
    imaginer = Imaginer(FakeFluxPolicy())
    recs = []
    for k, step in enumerate((0, 32)):
        (rec,) = imaginer.imagine(fake_batch(seed=k))
        rec.step, rec.index, rec.t_return = step, k, 100.0 + step / 30.0
        recs.append(rec)
    return recs


def frames(n, step0=0, t0=None):
    rng = np.random.default_rng(0)
    return [
        ActualFrame(rng.integers(0, 255, (64, 128, 3), dtype=np.uint8), step=None if t0 else step0 + i,
                    t=None if t0 is None else t0 + i / 30.0)
        for i in range(n)
    ]


def test_make_canvas_matches_side_by_side_layout():
    scene = np.zeros((480, 640, 3), np.uint8)
    wrist = np.full((480, 640, 3), 255, np.uint8)
    canvas = make_canvas(scene, wrist, (256, 512))
    assert canvas.shape == (256, 512, 3)
    assert canvas[:, :256].max() == 0 and canvas[:, 256:].min() == 255


def test_align_same_time_by_step(records):
    rows = align(records, frames(70))
    assert [r.chunk for r in rows[:2]] == [0, 0] and rows[32].chunk == 1
    assert [i for i, r in enumerate(rows) if r.chunk_start] == [0, 32]
    assert rows[0].offset == 0 and np.array_equal(rows[0].imagined, records[0].current_frame())
    assert np.array_equal(rows[10].imagined, records[0].frame_at(10))
    assert rows[10].surprise is not None and rows[0].surprise is None
    assert rows[69].offset == 37


def test_align_beyond_horizon_and_lookahead(records):
    rows = align(records[:1], frames(50))
    assert rows[45].imagined is None  # offset 45 > 42
    ahead = align(records, frames(40), mode="lookahead", lookahead=10)
    assert np.array_equal(ahead[5].imagined, records[0].frame_at(15))


def test_align_by_wall_clock(records):
    rows = align(records, frames(40, t0=100.0))
    assert rows[12].offset == 12 and rows[33].chunk == 1


def test_render_and_write(records, tmp_path):
    rows = align(records, frames(40))
    rendered = Renderer(task="pick").render_rows(rows)
    assert len(rendered) == 40 and rendered[0].ndim == 3
    path = write_gif(rendered[:5], tmp_path / "x.gif", fps=10)
    from PIL import Image

    assert Image.open(path).n_frames == 5
    sheet = contact_sheet(records[0], {o: f.image for o, f in enumerate(frames(43))})
    assert sheet.shape[0] > 2 * 128


def test_render_run_returns_surprise(records, tmp_path):
    surprise = render_run(records, frames(40), tmp_path / "run.gif", task="pick")
    assert surprise.shape == (40,) and np.isnan(surprise[0]) and np.isfinite(surprise[5])
    assert (tmp_path / "run.gif").exists()
