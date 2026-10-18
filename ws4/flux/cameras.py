"""Map robot cameras to a policy's camera keys (crop, mirror, resize) on the robot side.

The SO-101 package stretches each camera to 256x256 before tiling scene | wrist on a 256x512 canvas
(flux-action@830f1ce:docs/so101-lora.md:58-64; lerobot@146d847:src/lerobot/policies/flux3/f3/packing.py:162-171,
bilinear with antialias, no crop). Its training corpus was resized the same way (``camera_tile_hw`` 256x256 in
flux-action@830f1ce:configs/so101/community_corpus.json). We resize here, once, so transport stays small, and
send uint8 (BFL recommends uint8 or CPU float division: flux-action@830f1ce:docs/setup.md:220-221).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraView:
    """One model camera: which robot image feeds it and how to cut it.

    ``crop`` is a fractional box ``(x0, y0, x1, y1)`` in ``[0, 1]`` applied before resizing (for example the half
    of the head view in front of one arm). ``aspect`` center-crops to ``W / H`` first (for example 4/3 to match
    640x480 community cameras when the RealSense streams 16:9). ``mirror`` flips left/right; it only makes sense
    together with the matching joint sign flips (see ``ArmCalibration.mirrored``), and is an experiment.
    """

    source: str
    crop: tuple[float, float, float, float] | None = None
    aspect: float | None = None
    mirror: bool = False

    def __post_init__(self) -> None:
        if self.crop is not None:
            x0, y0, x1, y1 = self.crop
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError(f"invalid crop box {self.crop}")
        if self.aspect is not None and self.aspect <= 0:
            raise ValueError("aspect must be positive")


def _center_crop_to_aspect(img: np.ndarray, aspect: float) -> np.ndarray:
    h, w = img.shape[:2]
    if w / h > aspect:
        new_w = max(1, round(h * aspect))
        x0 = (w - new_w) // 2
        return img[:, x0 : x0 + new_w]
    new_h = max(1, round(w / aspect))
    y0 = (h - new_h) // 2
    return img[y0 : y0 + new_h]


def resize_uint8(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """HWC uint8 -> ``hw`` HWC uint8, bilinear (Pillow's bilinear widens its support when downscaling, which
    approximates torch's ``antialias=True``)."""
    h, w = hw
    if img.shape[:2] == (h, w):
        return np.ascontiguousarray(img)
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow ships with LeRobot; nearest-neighbour fallback otherwise
        ys = (np.arange(h) * img.shape[0] / h).astype(int)
        xs = (np.arange(w) * img.shape[1] / w).astype(int)
        return np.ascontiguousarray(img[ys][:, xs])
    return np.asarray(Image.fromarray(img).resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8)


def prepare_image(img: np.ndarray, view: CameraView, hw: tuple[int, int]) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{view.source}: expected HWC RGB, got {img.shape}")
    if img.dtype != np.uint8:
        if img.dtype.kind == "f" and img.max(initial=0) <= 1.0:
            img = img * 255.0
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    if view.aspect is not None:
        img = _center_crop_to_aspect(img, view.aspect)
    if view.crop is not None:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = view.crop
        top, left = round(y0 * h), round(x0 * w)
        img = img[top : max(round(y1 * h), top + 1), left : max(round(x1 * w), left + 1)]
    if view.mirror:
        img = img[:, ::-1]
    return resize_uint8(np.ascontiguousarray(img), hw)
