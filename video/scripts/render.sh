#!/usr/bin/env bash
# Render both formats, then finalize them into ../media/launch_video/.
# Run `python3 scripts/prepare_assets.py` once first (it builds public/ from ../media).
set -euo pipefail
cd "$(dirname "$0")/.."
# Prefer a preinstalled headless Chromium so rendering works offline.
for c in /opt/pw-browsers/chromium_headless_shell-*/chrome-linux/headless_shell; do
  [ -x "$c" ] && export REMOTION_CHROME="$c" && break
done
mkdir -p out ../media/launch_video
for comp in 16x9 9x16; do
  npx remotion render src/index.ts "Liminal$comp" "out/liminal_launch_$comp.raw.mp4" --concurrency="${CONCURRENCY:-4}"
  # Remotion encodes JPEG frames as full-range video; convert to standard TV range so every player shows it the same.
  ffmpeg -v error -y -i "out/liminal_launch_$comp.raw.mp4" \
    -vf "scale=in_range=pc:out_range=tv,format=yuv420p" -c:v libx264 -preset slow -crf 18 -profile:v high \
    -colorspace bt709 -color_primaries bt709 -color_trc bt709 -c:a aac -b:a 192k -movflags +faststart \
    "../media/launch_video/liminal_launch_$comp.mp4"
done
ls -lh ../media/launch_video
