"""Export independently verified teacher episodes through LeRobot's v3 writer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import JOINTS, TASK


def main(args):
    import numpy as np
    import torch
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    torch.set_num_threads(4)
    folders = []
    for split, source in [("train", args.train), ("validation", args.validation)]:
        if source is None:
            continue
        for path in sorted(Path(source).glob("episode-*/episode.npz")):
            report = json.loads((path.parent / "report.json").read_text())
            if report["success"]:
                folders.append((split, path, report))
    if args.limit is not None:
        folders = folders[: args.limit]
    if not folders:
        raise ValueError("No verified demonstrations")
    assert len({r["seed"] for _, _, r in folders}) == len(folders), "Duplicate episode seeds"
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(JOINTS)},
        "action": {"dtype": "float32", "shape": (6,), "names": list(JOINTS)},
        **{
            "observation.images." + role: {
                "dtype": "video",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            }
            for role in ("scene", "wrist")
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=Path(args.output),
        fps=args.fps,
        features=features,
        robot_type="so101_mujoco_" + args.frame,
        use_videos=True,
        image_writer_threads=4,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="ultrafast", crf=18, g=2),
    )
    provenance = []
    for index, (split, source, report) in enumerate(folders):
        with np.load(source, allow_pickle=False) as arrays:
            state = arrays["states"].astype(np.float32)
            action = arrays["actions"].astype(np.float32)
            if args.frame == "native":
                # Source recordings contain dataset90 absolute states and commands.
                for values in (state, action):
                    values[:, 1] = 90 - values[:, 1]
                    values[:, 2] -= 90
            cameras = {role: arrays["images." + role] for role in ("scene", "wrist")}
            assert state.shape == action.shape and state.shape[1:] == (6,)
            assert len(state) >= 50 and np.isfinite(state).all() and np.isfinite(action).all()
            assert report["max_lift_m"] > 0.04 and report["stable_inside_ticks"] >= 15
            for tick in range(len(state)):
                frame = {"observation.state": state[tick], "action": action[tick], "task": TASK}
                for role in ("scene", "wrist"):
                    image = cameras[role][tick]
                    assert image.shape == (256, 256, 3) and image.dtype == np.uint8
                    frame["observation.images." + role] = image
                dataset.add_frame(frame)
            dataset.save_episode()
        record = {
            "episode_index": index,
            "seed": report["seed"],
            "split": split,
            "frames": len(state),
            "source": str(source),
        }
        provenance.append(record)
        print(json.dumps(record), flush=True)
    dataset.finalize()
    output = Path(args.output)
    manifest = {
        "fps": args.fps,
        "task": TASK,
        "joint_order": list(JOINTS),
        "coordinate_mapping": (
            "native joint degrees; gripper 0-100 percent"
            if args.frame == "native"
            else "policy_shoulder=90-sim_shoulder; policy_elbow=sim_elbow+90"
        ),
        "normalization": args.normalization_description,
        "episodes": provenance,
        "total_frames": sum(x["frames"] for x in provenance),
        "heldout_policy": "validation episodes appended last for official per-task eval_split=0.2",
        "source": "scripted IK and contact physics; no policy-generated success labels",
    }
    (output / "simulation-provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("DATASET_READY", output, len(provenance), manifest["total_frames"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id", default="local/liminal_so101_sim_pick_place")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--frame", choices=["dataset90", "native"], default="dataset90")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--normalization-description", default="unchanged saved base checkpoint processors")
    main(parser.parse_args())
