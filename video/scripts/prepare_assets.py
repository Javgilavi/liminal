"""Cut, crop and upscale the repo media the video uses into public/ (regenerated, not committed).

Run from video/: python3 scripts/prepare_assets.py
Every source is a file under ../media or a render of the repo's MuJoCo twin. The twin render needs the twin's
venv (see ../examples/dual_so101_sim/README.md) and an EGL or OSMesa library.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
MEDIA = HERE.parent / "media"
PUBLIC = HERE / "public"

RED_PEN = MEDIA / "real_robot/flux_red_pen_runs/red_pen_insert_flux_lora_raw5k_layout_C_48s.mp4"
IMAGINATION = MEDIA / "real_robot/flux_imagination/red_pen_imagined_vs_actual_lora5k_ep101.mp4"
TWIN_PYTHON = HERE.parent / "examples/dual_so101_sim/.venv/bin/python"

# Upscale low-res robot cameras once, here, so the renderer only composites.
UPSCALE = "scale=iw*3:ih*3:flags=lanczos,unsharp=5:5:0.6:5:5:0.0"


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def encode(src: Path, out: str, vf: str, extra: tuple[str, ...] = ()) -> None:
    ff(*extra, "-i", str(src), "-vf", vf, "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "14",
       "-pix_fmt", "yuv420p", "-g", "15", str(PUBLIC / out))


def main() -> None:
    PUBLIC.mkdir(exist_ok=True)
    (PUBLIC / "fonts").mkdir(exist_ok=True)

    # The red-pen run is 640x240: head camera left, wrist camera right, burnt-in status bar at the bottom.
    encode(RED_PEN, "hero_head.mp4", f"crop=320:220:0:0,{UPSCALE}")
    encode(RED_PEN, "hero_wrist.mp4", f"crop=320:220:320:0,{UPSCALE}")
    wrist_raw = PUBLIC / "_wrist_raw.mp4"
    ff("-i", str(RED_PEN), "-vf", "crop=320:220:320:0", "-an", "-c:v", "libx264", "-crf", "10", str(wrist_raw))
    subprocess.run([sys.executable, "-I", str(HERE / "scripts/track_red_pen.py"), str(wrist_raw),
                    str(PUBLIC / "hero_wrist_track.json")], check=True)
    wrist_raw.unlink()

    # Imagined-vs-actual is 1048x340: [actual head | actual wrist | imagined head | imagined wrist].
    encode(IMAGINATION, "imag_actual.mp4", f"crop=254:254:266:31,{UPSCALE}")
    encode(IMAGINATION, "imag_imagined.mp4", f"crop=254:254:786:31,{UPSCALE}")

    for name, src in {
        "station.jpg": MEDIA / "real_robot/haller_xlerobot_station.jpg",
        # The team's front photo of the rig, background removed with BiRefNet (rembg birefnet-general).
        "robot_cutout.png": MEDIA / "real_robot/haller_rig_front_cutout.png",
        "rig_front.jpg": MEDIA / "real_robot/haller_rig_front.jpg",
    }.items():
        shutil.copy(src, PUBLIC / name)

    # History photos keep their own credits, see ../media/history/README.md.
    (PUBLIC / "history").mkdir(exist_ok=True)
    for photo in (MEDIA / "history").glob("*.jpg"):
        shutil.copy(photo, PUBLIC / "history" / photo.name)

    # Stock FLUX zero-shot strip: four 320x240 frames; drop the burnt-in status bar (shown as chips instead).
    for i in range(4):
        ff("-i", str(MEDIA / "real_robot/red_so101_flux_zero_shot_stack_cups_strip.png"), "-vf",
           f"crop=320:220:{i * 320}:0,scale=960:660:flags=lanczos", str(PUBLIC / f"strip_{i}.png"))

    fonts = HERE / "node_modules/@fontsource"
    for weight in (400, 500, 600, 700):
        shutil.copy(fonts / f"inter/files/inter-latin-{weight}-normal.woff2", PUBLIC / f"fonts/inter-{weight}.woff2")
    for weight in (400, 500):
        shutil.copy(fonts / f"jetbrains-mono/files/jetbrains-mono-latin-{weight}-normal.woff2",
                    PUBLIC / f"fonts/mono-{weight}.woff2")
    for style in ("normal", "italic"):
        shutil.copy(fonts / f"instrument-serif/files/instrument-serif-latin-400-{style}.woff2",
                    PUBLIC / f"fonts/serif-{style}.woff2")

    # The twin shot: about 15 min per format on a CPU renderer. Skipped when the frames already exist.
    for folder, w, h in (("twin_16x9", 1920, 1080), ("twin_9x16", 1080, 1920)):
        if not (PUBLIC / folder / "track.json").exists():
            cmd = [str(TWIN_PYTHON), str(HERE / "scripts/render_twin.py"), str(PUBLIC / folder), str(w), str(h)]
            subprocess.run(cmd, check=True, env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")})
    subprocess.run([sys.executable, "-I", str(HERE / "scripts/generate_music.py")], check=True)
    print("assets ready in", PUBLIC)


if __name__ == "__main__":
    main()
