"""Training on several data directories and the per-set, per-family evaluation of saved monitors."""

import json

from test_progress_pipeline import make_episode


def test_train_on_two_dirs_then_eval_sets_by_family(tmp_path):
    from ttt.eval_progress import family
    from ttt.eval_progress import main as eval_main
    from ttt.labels import EpisodeLabels
    from ttt.train_progress import is_test
    from ttt.train_progress import main as train_main

    names = [f"lid_cycles_2-{i:06d}" for i in range(8)]
    for i, name in enumerate(names):
        make_episode(tmp_path / ("a" if i % 2 else "b"), name=name, seed=i)
    # The closed loop names its tasks after the plan: they map back to the sim_episodes.py family.
    lab = EpisodeLabels.load(tmp_path / "b" / names[0] / "episode.json")
    lab.task = "flux_open_lid+close_lid+open_lid+close_lid"
    lab.save(tmp_path / "b" / names[0] / "episode.json")
    assert family(["open_lid", "close_lid"] * 2, lab.task) == "lid_cycles_2"
    assert family(["open_lid", "close_lid"], "x") == "lid_cycles_1" and family(["home"], "x") == "x"

    res = train_main(
        ["--data", str(tmp_path / "a"), str(tmp_path / "b"), "--memories", "gru", "--steps", "2", "--tag", "_mix",
         "--save-dir", str(tmp_path / "m")]
    )  # fmt: skip
    assert res["n_train"] + res["n_test"] == len(names) and len(res["data"]) == 2
    out = tmp_path / "r.json"
    ev = eval_main(
        ["--models", str(tmp_path / "m" / "progress_frame_mix_gru_s0.pt"), "--set", f"a={tmp_path / 'a'}",
         "--set", f"both={tmp_path}/*", "--all", "--out", str(out)]
    )  # fmt: skip
    r = ev["models"]["progress_frame_mix_gru_s0.pt"]
    assert ev["sets"]["a"]["episodes"] == 4 and ev["sets"]["both"]["episodes"] == len(names)
    assert set(r["both"]) == {"all", "lid_cycles_2"} and r["both"]["all"]["n"] == 12 * len(names)
    assert json.loads(out.read_text())["held_out_only"] is False
    held = eval_main(["--models", str(tmp_path / "m" / "progress_frame_mix_gru_s0.pt"), "--set", f"x={tmp_path}/*"])
    assert held["sets"]["x"]["episodes"] == sum(is_test(n) for n in names)
