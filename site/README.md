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
