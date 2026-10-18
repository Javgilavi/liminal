"""Cut, crop and upscale the repo media the video uses into public/ (regenerated, not committed).

Run from video/: python3 scripts/prepare_assets.py
Every source is a file under ../media, so the video only shows footage the repo documents.
"""

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
MEDIA = HERE.parent / "media"
PUBLIC = HERE / "public"

RED_PEN = MEDIA / "real_robot/flux_red_pen_runs/red_pen_insert_flux_lora_raw5k_layout_C_48s.mp4"
IMAGINATION = MEDIA / "real_robot/flux_imagination/red_pen_imagined_vs_actual_lora5k_ep101.mp4"
LIVE_UI = MEDIA / "liminal_ui/live_ui_flux_mujoco_session.mp4"
REAL_VS_SIM = MEDIA / "digital_twin/real_vs_sim/haller_photo_vs_sim.jpg"

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

    # Live UI: real FLUX inference on simulated arms. Keep the reactive task tree panel.
    ff("-ss", "22", "-i", str(LIVE_UI), "-frames:v", "1", "-vf", "crop=1010:655:410:345",
       str(PUBLIC / "ui_task_tree.png"))

    # haller_photo_vs_sim.jpg is a 2x4 grid of 640x360 tiles; the top row is the real photo and its twin.
    ff("-i", str(REAL_VS_SIM), "-vf", "crop=592:333:24:30", str(PUBLIC / "twin_real.png"))
    ff("-i", str(REAL_VS_SIM), "-vf", "crop=592:333:664:30", str(PUBLIC / "twin_sim.png"))

    for name, src in {
        "gripper.jpg": MEDIA / "real_robot/haller_gripper_closeup.jpg",
        "station.jpg": MEDIA / "real_robot/haller_xlerobot_station.jpg",
    }.items():
        shutil.copy(src, PUBLIC / name)

    fonts = HERE / "node_modules/@fontsource"
    for weight in (400, 500, 600, 700):
        shutil.copy(fonts / f"inter/files/inter-latin-{weight}-normal.woff2", PUBLIC / f"fonts/inter-{weight}.woff2")
    for weight in (400, 500):
        shutil.copy(fonts / f"jetbrains-mono/files/jetbrains-mono-latin-{weight}-normal.woff2",
                    PUBLIC / f"fonts/mono-{weight}.woff2")
    subprocess.run([sys.executable, "-I", str(HERE / "scripts/generate_music.py")], check=True)
    print("assets ready in", PUBLIC)


if __name__ == "__main__":
    main()
