# Liminal research page and measured-joint twin

**Liminal** is the project; **Haller** is the hardware platform and its controller.

A local paper-style demonstration page, interactive 3D replay, and live operator companion. It is an additional entry point; the existing `liminal serve` operator UI and official inference CLI remain available.

```bash
# From the source checkout, with UI/live/MuJoCo dependencies installed:
uv pip install --python .venv/bin/python -e '.[research]'
./scripts/research_ui.sh
# http://127.0.0.1:18003/
```

The initial page is observation-only and can be explored with Haller disconnected. Its interactive replay comes from one saved **scripted MuJoCo demonstration**, not learned or real-robot execution. Both robot arms, the lid and movable objects use recorded poses. Play, pause, scrub, select a camera preset, drag to orbit and scroll to zoom. The displayed source photograph comes from the user's setup recording.

The launcher enables pairing to the project GitHub Pages site by default; pass
`--site-url` to use another trusted HTTPS deployment. Open
**http://127.0.0.1:18003/pair** to connect the full-screen public lab. Viewing
measured joints and cameras requires no policy profile and does not acquire a
policy connection. The pairing page shows controller availability and links to
Haller. Pairing while Haller is offline is supported; observation reconnects
automatically when it becomes available.

## Connect the existing robot owner

Keep Haller running at `http://127.0.0.1:8000`. The companion subscribes to `/ws/telemetry` and proxies existing `/cameras/{top,wrist_a,wrist_b}/snapshot` images. It never opens serial ports, a V4L device, or Haller's exclusive policy transport merely to show the page. An existing CLI policy can continue while the companion observes it. With a controller on another origin, `--controller http://HOST:PORT` supports read-only observation; use an SSH tunnel for another machine.

For prompt-triggered inference, supply the same verified profile used by the official runner:

```bash
./scripts/research_ui.sh --profile configs/policy/official_haller_right.yaml
# If using the hardware environment:
LIMINAL_UI_PYTHON=.venv-robot/bin/python ./scripts/research_ui.sh \
  --profile configs/policy/official_haller_right.yaml
```

The profile determines the GPU transport and checkpoint. The current main profile uses the existing GCP/IAP worker; this UI does not silently switch it to Runpod. Adapt the profile/transport separately if the commissioned hardware now uses a different worker.

Submitting a prompt calls the existing `liminal.policy.live.run`, preserving its `SafetySupervisor`, camera/unit verification, calibration travel checks, acknowledgements, stall/load checks and cleanup. Default prediction is observation-only. For physical execution, select **Execute on the real robot**, manually set the selected arm to MANUAL/engaged in Haller, and check that acknowledgement. The UI does not engage torque or clear STOP. A profile selects one policy arm; both arms can still be mirrored. A twelve-channel learned dual-arm policy is a separate capability.

One trial may run from this companion at a time. An arm reporting policy ownership is refused, and Haller's policy handshake remains authoritative for conflicts with external runners. **Stop this trial** cancels only a trial launched here, allowing the existing runner's cleanup to request stop/release. Haller remains the hardware stop interface. Trial files are saved under `runs/research/` on this machine. They include the official runner's observations, plans and reports; a completed runner duration is not a manipulation success label.

## What the twin does

The server applies measured native joint degrees and gripper percentages directly to the existing SO-101 CAD and calls `mj_forward`. It never applies a policy's historical-coordinate or normalization transform to measured robot positions, and does not clip readings to approximate CAD joint limits. The browser receives body transforms and renders the actual compiled mesh geometry with locally vendored Three.js. Orbiting changes only the view.

Live rendering uses 65 ms visual interpolation between observed poses and freezes when telemetry becomes stale. The 30 Hz websocket is a delivery cadence, not a claim that the controller samples at 30 Hz. The displayed observed rate and sample age come from actual arrivals. Samples older than 500 ms disable trial launch. If the controller supplies `t_ms`, stale source timestamps are also rejected; clocks need synchronization.

Only arm poses are mirrored from live telemetry. Table, markers and box are an **estimated** reconstruction; live object poses and contact forces are not tracked. Real camera intrinsics/extrinsics, base spacing, finger compliance and coordinate registration need physical calibration before calling this a precise physical twin. Live camera snapshots are separate from the 3D renderer and are not substituted with the recorded demonstration.

## Implementation and validation

- `src/liminal/ui/research.py`: companion API, telemetry observer and guarded trial launch.
- `src/liminal/ui/twin.py`: read-only kinematics and browser geometry export; optional MuJoCo imports stay lazy.
- `src/liminal/ui/static/research/`: buildless page, self-hosted fonts/Three.js, compact simulated replay and demo video.
- `examples/dual_so101_sim/research_replay/`: estimated scene snapshot used by the packaged replay and joint mirror.
- `scripts/build_research_replay.py --source PATH_TO_TRAJECTORY.npz`: regenerate compact replay transforms.

Contract tests use fake telemetry, a fake mirror and a mocked official runner: no hardware, GPU, MuJoCo or real network. Browser verification uses the existing local environment and a headless browser. Screenshots are generated in ignored `outputs/research-ui/`.

The presentation references [DexMimic](https://zhexiluo.github.io/DexMimic/) and [Nerfies](https://nerfies.github.io/), with original Liminal layout, copy, controls and styles. No reference-site source or research claims are copied. Three.js 0.180.0 and its license are vendored; DM Sans and Instrument Serif are self-hosted with their font licenses.

## Public GitHub Pages project and full-screen lab

The static project lives in `site/` and is deployed from the repository's dedicated
`gh-pages` branch at **https://javgilavi.github.io/liminal/**. Its separate lab is
**https://javgilavi.github.io/liminal/live.html**. This does not change the existing
personal GitHub Pages site or make the source repository public.

The public page summarizes main revision `07e50d0`, including the corrected
five-seed simulation comparison, newer real red-pen insertion highlights, older official hardware trial index and separate twin
media archive. Result domains and simulation/scripted/predicted evidence labels
stay separate. `site/evidence/results.json` is an explicit aggregate export;
raw logs, calibration profiles and machine identifiers are excluded.

To pair the public lab with this companion:

```bash
./scripts/research_ui.sh \
  --profile configs/policy/official_haller_right.yaml \
  --site-url https://javgilavi.github.io/liminal/
# Open http://127.0.0.1:18003/pair on the same machine.
```

The pairing page provides a private link back to the public lab. Its token is in
an HTTPS URL fragment, removed immediately by the browser, and retained only in
memory. The token expires on companion restart. Cross-origin API requests need
both the configured site origin and bearer token; the websocket authenticates
with a subprotocol token. Pairing itself is loopback-only, non-cacheable and
cannot be embedded. Same-origin local UI use remains available. Browser local
network permissions may require an explicit allow decision; the site does not
expose Haller publicly. Live frames and telemetry travel directly to the browser,
not through GitHub. Disconnecting the browser does not stop a local trial.

On pairing, the public lab fetches `/api/scene` from the companion and rebuilds
its viewer using that exact compiled model. Mesh identifiers are scoped to each
model, so a new scene cannot accidentally reuse the public replay's geometry.
Disconnect restores the saved public scene. If the websocket is unavailable,
authenticated HTTP polling supplies fresh snapshots while it reconnects. Camera
labels use the profile's verified wrist binding when available; wrist A is not
assumed to be the physical left arm. Camera switching discards in-flight frames
from the previous camera or pairing session.

Source freshness accepts Haller's Unix-seconds `t` and legacy `t_ms`, rejecting
nonfinite or >500 ms clock differences. Sample age includes the source's age at
arrival plus elapsed time since receipt. Missing policy-arm telemetry blocks
trial launch even if the other arm remains connected.

The public replay has scene, left-wrist and right-wrist views, a timeline and
rotatable CAD. It does not send an instruction to a policy until paired, and
hardware execution remains separately gated by the existing runner. The public
lab displays incoming poses at the observed cadence and freezes on stale data;
the original local page additionally interpolates visually over 65 ms.

To stage a reviewed public export without modifying a source checkout:

```bash
.venv/bin/python scripts/export_project_site.py --output /tmp/liminal-public-site
```

This copies only the known static-site files and assets and checks local links.
Publication uses a separate deployment checkout, preserving unrelated source
changes. The website has no analytics, remote fonts or third-party embed scripts.

### Loading and device fallbacks

Project navigation, result tabs, recordings and local pairing initialize without
waiting for the optional 3D modules. On the project page, desktop scenes load
progressively; phones start with a recorded scene preview and an explicit **Explore in 3D** control.
Each viewer can fail or lose its graphics context independently, retaining a
preview and a retry control. Camera views remain separate from these previews.
Recording and diagnostic selectors use native menus on small screens; desktop
views retain keyboard-accessible tabs. Missing replay/chart data disables the
unavailable control and offers recovery without breaking the rest of the page.
