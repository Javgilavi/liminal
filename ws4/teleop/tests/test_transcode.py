import json

import pytest
import transcode
import validate
from conftest import make_haller_dataset


def test_h264_720p_recording_becomes_mergeable_av1(tmp_path, calib_dir):
    av = pytest.importorskip("av")
    try:
        av.codec.Codec("libsvtav1", "w")
    except Exception:
        pytest.skip("PyAV build without libsvtav1")
    src = make_haller_dataset(tmp_path / "src", codec="h264", hw=(240, 320), n_eps=1, ep_len=12)
    assert any("codec h264" in e for e in validate.validate(src, None, calib_dir).errors)
    dst = tmp_path / "dst"
    counts = transcode.transcode(src, dst, size=(480, 640))
    assert set(counts.values()) == {12}
    info = json.loads((dst / "meta/info.json").read_text())
    head = info["features"]["observation.images.head"]
    assert head["shape"] == [480, 640, 3] and head["info"]["video.codec"] == "av1"
    rep = validate.validate(dst, None, calib_dir)
    assert rep.errors == [], rep.errors
    with av.open(str(dst / "videos/observation.images.head/chunk-000/file-000.mp4")) as c:
        s = c.streams.video[0]
        assert s.codec_context.name in ("libdav1d", "libaom-av1", "av1", "libsvtav1") or "av1" in s.codec_context.name
        assert (s.codec_context.height, s.codec_context.width) == (480, 640)
