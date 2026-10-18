"""Export native/HF/20 Hz atomic skills using the official LeRobot v3 writer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scene import FPS, PROMPTS


def main(root: Path):
    import numpy as np
    import torch
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    torch.set_num_threads(4)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        },
        **{
            f"observation.images.{role}": {
                "dtype": "video",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            }
            for role in ["scene", "wrist"]
        },
    }
    dataset = LeRobotDataset.create(
        repo_id="local/liminal_nist_atomic_assembly",
        root=root / "dataset-v3",
        fps=FPS,
        features=features,
        robot_type="so101_mujoco_native",
        use_videos=True,
        image_writer_threads=4,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="ultrafast", crf=18, g=2),
    )
    provenance = []
    for split in ["train", "validation"]:
        files = sorted((root / "raw" / split).glob("episode-*/report.json"))
        for file in files:
            report = json.loads(file.read_text())
            if not report["success"]:
                continue
            assert len(report["segments"]) == 4 and all(segment["success"] for segment in report["segments"])
            with np.load(file.parent / "episode.npz") as arrays:
                # NPZ fields decompress on each lookup; materialize once per
                # assembly instead of decompressing full camera arrays per tick.
                states = arrays["states"].astype(np.float32)
                actions = arrays["actions"].astype(np.float32)
                skills = arrays["skill"]
                cameras = {role: arrays[f"images.{role}"] for role in ["scene", "wrist"]}
                for segment in report["segments"]:
                    start, end = segment["start"], segment["end"]
                    assert end - start >= 50
                    assert set(skills[start:end]) == {segment["skill"]}
                    for tick in range(start, end):
                        frame = {
                            "observation.state": states[tick],
                            "action": actions[tick],
                            "task": PROMPTS[segment["skill"]],
                        }
                        for role in ["scene", "wrist"]:
                            frame[f"observation.images.{role}"] = cameras[role][tick]
                        dataset.add_frame(frame)
                    dataset.save_episode()
                    row = {
                        "episode_index": len(provenance),
                        "split": split,
                        "seed": report["metadata"]["seed"],
                        "skill": segment["skill"],
                        "frames": end - start,
                        "source": str(file.parent / "episode.npz"),
                        "start": start,
                        "end": end,
                    }
                    provenance.append(row)
                    print(json.dumps(row), flush=True)
    dataset.finalize()
    manifest = {
        "fps": FPS,
        "coordinate_frame": "native",
        "normalization": "HF simulator, saved before training",
        "tasks": PROMPTS,
        "episodes": provenance,
        "frames": sum(row["frames"] for row in provenance),
        "source": "complete contact-verified scripted assemblies segmented at atomic predicates",
    }
    (root / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
