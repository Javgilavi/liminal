"""Does the FLUX runner see what the recorder recorded? Compare a recorded dataset frame with a live frame read through
the runner's camera path (LeRobot ``OpenCVCamera``, same config as ``run_single_arm.py``).

Reports per-channel means (colour), brightness, the mean absolute difference, and whether the live frame matches
better as is or rotated 180 degrees (orientation). Put the arm in the pose of the recorded frame first (episode start
= rest pose). Opens cameras only, never the arm ports, and only after the HMI is stopped (it holds the cameras too).

    python ws4/single_arm/compare_frames.py --root <dataset dir> --repo-id <name> --episode 0 --frame 0 \\
        --key observation.images.right_wrist --device /dev/v4l/by-path/pci-0000:00:14.0-usb-0:7.2:1.0-video-index0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def recorded_frame(repo_id: str, root: str | None, episode: int, frame: int, key: str) -> np.ndarray:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root, episodes=[episode])
    item = ds[frame]  # with an episode filter, index 0 is that episode's first frame (UNVERIFIED on v3 multi-episode)
    if int(item["episode_index"]) != episode:
        raise SystemExit(f"index {frame} is episode {int(item['episode_index'])}, not {episode}")
    img = item[key]  # (3, H, W) float in [0, 1]
    return (img.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)


def live_frame(device: str, fourcc: str | None, rotate: int) -> np.ndarray:
    from lerobot.cameras.configs import Cv2Rotation
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    rotation = {0: Cv2Rotation.NO_ROTATION, 180: Cv2Rotation.ROTATE_180}[rotate]
    cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=Path(device), fps=30, width=640, height=480, fourcc=fourcc,
                                          rotation=rotation, warmup_s=3))  # fmt: skip
    cam.connect()
    try:
        for _ in range(10):
            frame = cam.async_read(timeout_ms=500)
        return np.asarray(frame)
    finally:
        cam.disconnect()


def stats(img: np.ndarray) -> str:
    m = img.reshape(-1, 3).mean(0)
    return f"RGB means {m.round(1)}, brightness {img.mean():.1f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--key", required=True, help="dataset camera key, e.g. observation.images.head")
    p.add_argument("--device", required=True, help="camera path as the runner opens it")
    p.add_argument("--fourcc", default="none")
    p.add_argument("--rotate", type=int, choices=(0, 180), default=0, help="as the runner: 180 for the Haller wrist")
    p.add_argument("--out", default=str(HERE / "frames" / "haller" / "parity.jpg"))
    args = p.parse_args()
    import cv2

    rec = recorded_frame(args.repo_id, args.root, args.episode, args.frame, args.key)
    live = live_frame(args.device, None if args.fourcc.lower() == "none" else args.fourcc, args.rotate)
    if live.shape != rec.shape:
        live = cv2.resize(live, (rec.shape[1], rec.shape[0]))
    as_is = np.abs(live.astype(float) - rec).mean()
    rotated = np.abs(np.rot90(live, 2).astype(float) - rec).mean()
    print(f"recorded {args.key}: {stats(rec)}")
    print(f"live     {args.device}: {stats(live)}")
    print(f"mean |live - recorded| {as_is:.1f}; with the live frame rotated 180 deg {rotated:.1f} "
          f"-> {'orientation matches' if as_is <= rotated else 'ROTATED: the runner must rotate 180'}")  # fmt: skip
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(np.concatenate([rec, live], axis=1), cv2.COLOR_RGB2BGR))
    print(f"saved {args.out} (recorded | live)")


if __name__ == "__main__":
    main()
