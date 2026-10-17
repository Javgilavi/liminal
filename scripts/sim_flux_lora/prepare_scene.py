"""Materialize the versioned single-arm scene against a local Menagerie asset checkout."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

MENAGERIE_REVISION = "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--menagerie", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    checkout = args.menagerie.resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != MENAGERIE_REVISION:
        raise ValueError(f"expected Menagerie {MENAGERIE_REVISION}, got {revision}")
    assets = checkout / "robotstudio_so101/assets"
    if not assets.is_dir():
        raise FileNotFoundError(assets)
    source = Path(__file__).resolve().parents[2] / "sim/single_arm"
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("scene.xml", "so101.xml", "official_camera_setup.py", "LICENSE"):
        shutil.copy2(source / name, args.output / name)
    (args.output / "assets").symlink_to(assets, target_is_directory=True)
    (args.output / "asset-source.json").write_text(json.dumps({
        "repository": "https://github.com/google-deepmind/mujoco_menagerie",
        "revision": revision,
        "assets": str(assets),
    }, indent=2) + "\n")
    print(f"export LIMINAL_SIM_ROOT={args.output.resolve()}")


if __name__ == "__main__":
    main()
