"""Render generated twin demonstrations into a LeRobot v3 dataset for FLUX LoRA training.

Adapted from Codex's training/export_dataset.py: replay each episode's recorded qpos in the same
randomized scene, render the D455 (`top` -> observation.images.scene) and the active arm's wrist camera
(-> observation.images.wrist) at 256x256, and convert joints to the checkpoint's coordinates
(shoulder +180, elbow +90; applied exactly once here). Run where LeRobot is installed (the Pod).

usage: MUJOCO_GL=egl python -m twin.export --raw DIR --out DIR [--repo-id local/haller-twin-v1] [--media DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import av  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from .scene import Layout, build  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def to_checkpoint(values):
    a = np.asarray(values, np.float32).copy()
    a[1] += 180.0
    a[2] += 90.0
    return a


def write_mp4(path, frames, fps=30):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt, stream.options = "yuv420p", {"crf": "23", "preset": "fast"}
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main():
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo-id", default="local/haller-twin-v1")
    ap.add_argument("--media", type=Path, help="optional dir for a few front-view preview MP4s")
    args = ap.parse_args()
    assert not args.out.exists(), "never overwrite an exported dataset"
    episodes = sorted(p for p in args.raw.glob("*.npz"))
    # Interleave tasks so the trainer's last-20 %-per-task hold-out sees every task.
    features = {k: {"dtype": "float32", "shape": (6,), "names": JOINTS} for k in ("observation.state", "action")}
    features.update({k: {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channels"]}
                     for k in ("observation.images.scene", "observation.images.wrist")})
    dataset = LeRobotDataset.create(args.repo_id, 30, features, root=args.out, robot_type="so101_follower",
                                    streaming_encoding=True, rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=20, preset="fast", g=15),
                                    encoder_threads=2, video_backend="pyav")
    if args.media:
        args.media.mkdir(parents=True, exist_ok=True)
    t0, audit = time.monotonic(), []
    for index, path in enumerate(episodes):
        meta = json.loads(path.with_suffix(".json").read_text())
        result, lay = meta["result"], meta["layout"]
        assert result["success"]
        layout = Layout(box=tuple(lay["box"]), lid_deg=lay["lid_deg"], objects={k: tuple(v) for k, v in lay["objects"].items()},
                        light_scale=lay["light_scale"], camera_jitter=tuple(lay["camera_jitter"]))
        model, data = build(layout)
        model.vis.quality.shadowsize = 1024
        ren = {"top": mujoco.Renderer(model, 240, 320), "wrist": mujoco.Renderer(model, 180, 320)}
        front = mujoco.Renderer(model, 360, 480) if args.media and index < 6 else None
        wrist_cam = f"{result['arm']}_wrist"
        npz = np.load(path)
        rows, qobs = npz["rows"], npz["qobs"]
        preview = []
        for row, q in zip(rows, qobs):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            ren["top"].update_scene(data, camera="top")
            scene = np.asarray(Image.fromarray(ren["top"].render()).resize((256, 256), Image.Resampling.BILINEAR))
            ren["wrist"].update_scene(data, camera=wrist_cam)
            wrist = np.asarray(Image.fromarray(ren["wrist"].render()).resize((256, 256), Image.Resampling.BILINEAR))
            dataset.add_frame({"observation.state": to_checkpoint(row[1:7]), "action": to_checkpoint(row[7:13]),
                               "observation.images.scene": scene, "observation.images.wrist": wrist, "task": result["prompt"]})
            if front is not None:
                front.update_scene(data, camera="front")
                preview.append(np.hstack([front.render(), np.asarray(Image.fromarray(np.vstack([scene, wrist])).resize((180, 360)))]))
        dataset.save_episode(parallel_encoding=False)
        if preview:
            write_mp4(args.media / f"{path.stem}.mp4", preview)
        for r in ren.values():
            r.close()
        if front is not None:
            front.close()
        audit.append({"dataset_episode": index, "source": path.stem, "task": result["task"], "prompt": result["prompt"], "frames": len(rows)})
        print(json.dumps({"episode": index, "source": path.stem, "frames": len(rows), "elapsed_s": round(time.monotonic() - t0, 1)}), flush=True)
    dataset.finalize()
    summary = {"episodes": audit, "total_frames": sum(a["frames"] for a in audit),
               "valid_unpadded_windows": sum(max(0, a["frames"] - 49) for a in audit),
               "tasks": sorted({a["prompt"] for a in audit}), "fps": 30,
               "camera_order": ["scene=top (D455)", "wrist=active arm"],
               "state_action_units": "first five: checkpoint degrees (shoulder +180, elbow +90); gripper 0-100",
               "domain": "simulation twin with estimated geometry; scripted expert with closed-loop corrections",
               "training_ready": True, "elapsed_s": round(time.monotonic() - t0, 1)}
    (args.out / "demonstration_audit.json").write_text(json.dumps(summary, indent=2))
    print("EXPORT_COMPLETE", json.dumps({k: summary[k] for k in ("total_frames", "tasks")}), flush=True)


if __name__ == "__main__":
    main()
