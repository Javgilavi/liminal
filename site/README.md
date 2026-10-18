# Liminal public project page

Published at https://javgilavi.github.io/liminal/ from the `gh-pages` branch.
`live.html` is the separate full-screen lab, with browser-only recorded replay
and explicit pairing to the local research companion.

No build framework or server runtime is required for the website. Serve this
folder with a static HTTP server for preview. Three.js 0.180.0 and fonts are
self-hosted with their licenses. Relative asset paths support GitHub project Pages.

Content is based on source main `07e50d0985bb91c4f613017df7a30e9ea68394c0`.
See `evidence/notes.html` and `evidence/results.json` for experiment provenance,
limitations and bridge setup. Publishing this static tree does not publish raw
logs, profiles, credentials or the private source repository.

The compact scene is the compiled research-replay MuJoCo scene; binary mesh
buffers are gzip compressed. Its poses and three camera videos come from one
scripted 14-second expert trajectory, with the left arm parked and lid fixed.
This is not learned-policy task success. Selected checkpoint and diagnostic
videos come from the source repository's labeled `media/` archive.

Real-run website copies are encoded as H.264/yuv420p with fast-start metadata; original MP4V recordings in `media/` remain intact.

Drive media review, 18 October 2026: two gallery screen recordings were replaced
with the team's 60-second project film and an outside view of the cutter trial.
The five red-pen camera recordings and matched simulation comparisons are unchanged.
Sources in the team Drive linked from `media/README.md`:

- `05_video/liminal_60s_v1.mp4` → `assets/liminal-overview.mp4`
  (full 60 seconds; original audio and the film's speed/replay labels retained).
- `20261018_093134.mp4` → `assets/real-cutter-trial.mp4`
  (full 32.4 seconds at recorded speed; ambient audio omitted).

The cutter recording shows grasp and carry toward the box, without a completed
release. It is presented as a recorded trial, not an additional scored success.
Both website copies use H.264/yuv420p and fast-start metadata. Their posters are
frames from those same clips, at 6 seconds and 14 seconds respectively.
