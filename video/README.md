# Liminal launch video

A 48-second product-style showcase, built in code with [Remotion](https://www.remotion.dev/).
Renders: [`media/launch_video/liminal_launch_16x9.mp4`](../media/launch_video/liminal_launch_16x9.mp4)
(1920x1080) and [`liminal_launch_9x16.mp4`](../media/launch_video/liminal_launch_9x16.mp4) (1080x1920),
30 fps, H.264 + AAC, original synthesized score.

Each shot is one of: repo footage under `../media`, a render of the repo's MuJoCo twin, a line-art
illustration, or a motion graphic that states a fact from the repo. Simulated footage always carries the
filled `SIM` tag; real footage carries `REAL`; FLUX's predicted video is tagged `IMAGINED · FLUX`.

## Edit and re-render

```bash
cd video
npm ci
python3 scripts/prepare_assets.py   # crops ../media into public/, tracks the pen, renders the twin, writes music.wav
npm run studio                       # live preview
bash scripts/render.sh               # renders both formats into ../media/launch_video/
```

`prepare_assets.py` renders the twin shot with the twin's own venv
(`uv venv ../examples/dual_so101_sim/.venv && uv pip install --python ../examples/dual_so101_sim/.venv/bin/python mujoco pillow numpy`)
and needs an EGL or OSMesa library (`apt-get install libegl1 libosmesa6`). It takes about 15 minutes
per format on a CPU and is skipped when `public/twin_*/track.json` exists.

All on-screen text, colours, the grade, shot timings, the book's pages, the reveal callouts, hero speed
ramps and the diagram layout live in [`config.json`](config.json). Keep cuts on the 0.5 s beat grid.
Set `"music": ""` to render silent, or drop any 120 BPM track in `public/` and point `music` at it.

| File | Role |
|---|---|
| `config.json` | Text, colours, grade, timings, book pages, callouts, diagram |
| `scripts/prepare_assets.py` | Builds `public/` from `../media` (crops, 3x Lanczos upscale, cutout, twin render, music) |
| `scripts/render_twin.py` | The scripted dual-arm shot in the MuJoCo twin, plus per-frame 2D gripper positions for the HUD |
| `scripts/track_red_pen.py` | Finds the red pen cap per frame in the wrist camera; the HUD box follows it |
| `scripts/generate_music.py` | Synthesizes the original 120 BPM score, page-flip sounds included |
| `scripts/stills.mjs` | Review stills: `node scripts/stills.mjs Liminal16x9 /tmp/out 7.5 14` |
| `src/scenes/Book.tsx`, `src/components/Art.tsx` | The history book and its line-art plates |
| `src/scenes/Reveal.tsx`, `Sim.tsx`, `Hero.tsx`, `Middle.tsx`, `Closing.tsx` | The remaining shots |
| `src/components/ui.tsx` | Shared cards, tags, chips, HUD brackets, wordmark, transitions, grain |

## Storyboard (v2, 48 s)

Beat grid: 120 BPM, one beat = 0.5 s, bar = 2 s. Every cut lands on a beat.

| Time | Shot | Source | Tag |
|---|---|---|---|
| 0.0-2.0 | A closed book rises out of the dark: *Robot Arms, a short history, 1961-2026* | — | — |
| 2.0-13.2 | Cover opens; a page turns every 1.5 s; each plate draws itself (see the history list below) | `Art.tsx` line art | illustration |
| 12.5-14.0 | Final spread "2026 · Liminal"; the camera pans onto the plate and dives into it | `haller_rig_front_cutout.png` | REAL |
| 14.0-16.2 | Match cut to the real rig, lit like a product shot; callouts: D455, both SO-101 arms, wrist camera, XLeRobot tower | same cutout | REAL |
| 16.2-18.0 | Rig moves aside, LIMINAL wordmark, "Dual-arm orchestration for FLUX 3 Action" | — | — |
| 18.0-19.0 | A scan line turns the real rig into its MuJoCo twin | `render_twin.py` | SIM |
| 19.0-20.6 | "Some tasks need two hands." | twin render | SIM |
| 20.6-26.0 | Left arm `open_lid`, then holds it (`while maintain lid_open`) while the right arm `pick_place`s the red marker; HUD brackets track both grippers; the task-tree panel shows `configs/tasks/lid_hold_insert.yaml` live | twin render + `track.json` | SIM |
| 26.0-29.0 | Real robot, head + wrist cards, 4x | red-pen run layout C, 2.0-14.0 s | REAL |
| 29.0-34.5 | Speed ramp to 2x; slot target; `red_pen_in_slot` ✓ at source 38.0 s; "3 of 5 · team-reported" | layout C, 29.0-40.0 s | REAL |
| 34.5-36.0 | Imagined vs actual: "Imagine, then act." | `…lora5k_ep101.mp4` 4.6-6.1 s | model / REAL |
| 36.0-40.5 | Architecture diagram draws on | `config.json` | — |
| 40.5-43.5 | Terminal: quick start and its real output; "Runs on mock arms. No hardware." | README | — |
| 43.5-48.0 | End card | — | — |

Music impacts: 14.0, 18.0, 26.0, 36.0, 43.5 s. Risers lead into 14.0, 18.0, 26.0 and 43.5. A page-flip
sound plays on every turn; a soft pulse enters at the 2023 pages; drums drop for 34.5-36.0 and the end card.

### History pages: facts to double-check before publishing

These come from general knowledge, not from the repo.

| Year | Robot | Line on the page | Who |
|---|---|---|---|
| 1961 | Unimate | The first industrial robot arm. | Unimation, installed at General Motors |
| 1969 | Stanford Arm | Electric, and run by a computer. | Victor Scheinman, Stanford |
| 1978 | PUMA | A programmable arm for assembly. | Unimation |
| 2012 | Baxter | Two arms, safe beside people. | Rethink Robotics |
| 2023 | ALOHA | Low-cost arms learn two-handed skills. | Stanford |
| 2023 | RT-2 | Vision and language become actions. | Google DeepMind |
| 2024 | SO-100 · SO-101 | Open-source arms anyone can print. | TheRobotStudio, Hugging Face (SO-101 followed in 2025) |

The plates are stylized drawings, labelled "Illustrations, not to scale" on screen.

### The twin shot

`render_twin.py` builds the repo's Haller twin (`examples/dual_so101_sim/twin`) with dark studio lighting
and an orbiting camera. Joint targets come from the twin's own expert planner: `open_box_waypoints`
for the left arm (stopped while it still holds the lid at 80°) and `pick_place_waypoints` for the red
marker on the right. Motion is keyframed between those IK solutions, and the marker is carried
kinematically. It illustrates the task; it is **not** a physics or policy rollout, and the frame says so.
The right arm starts only once the lid is open, as the `while` node in `lid_hold_insert.yaml` requires.
The HUD uses the twin expert's own success threshold (lid > 65°) for `open_lid`.

## Asset notes

Tags: **REAL POV** robot cameras, **REAL RIG** photos and phone video of the rig, **SIM**, **UI**.

| Asset | Tag | Size, length | What it shows / best moments |
|---|---|---|---|
| `real_robot/haller_rig_front.jpg` | REAL RIG | 2040x1530 | **Used** (as the cutout). The team's front photo: tower, D455, both arms, box, markers, cutter. |
| `real_robot/haller_rig_front_cutout.png` | REAL RIG | 830x900 RGBA | The rig from that photo, background removed with BiRefNet (`rembg`, `birefnet-general`). |
| `real_robot/flux_red_pen_runs/…layout_C_48s.mp4` | REAL POV | 640x240 (2x 320x240), 47.9 s | **Used.** Head + wrist cams. Grasp 2-14 s; pen lowered into slot 29-37 s; released 37-39 s; a hand resets the scene from 42 s. |
| `…layout_A_94s.mp4`, `…layout_B_91s.mp4`, `…morning_*` | REAL POV | 640x240 | Same task; layout C has the cleanest insert. |
| `real_robot/flux_imagination/…lora5k_ep101.mp4` | REAL POV + model | 1048x340, 9.9 s | **Used.** Actual vs imagined, held-out take 101. |
| `real_robot/haller_xlerobot_station.jpg`, `haller_gripper_closeup.jpg` | REAL RIG | 1600x1200, 1512x2016 | v1's hook and reveal; replaced by the front photo. |
| `real_robot/haller_tabletop_real_scene_2026-10-18.mp4` | REAL RIG | 478x850, 20.4 s | Handheld walk around the table; too shaky for this cut. |
| `digital_twin/*`, `flux_sim/*` | SIM | up to 800x600 | Earlier twin and zero-shot renders; the v2 twin shot is rendered fresh at full resolution. |
| `liminal_ui/*` | UI | 1440 wide | Not used in v2. |
| Drive `IMG_6374/6390/6397.MOV` | REAL RIG | 28-164 MB | Listed, not downloaded: Drive downloads are blocked in this environment. |

## Choices to know about

- **Arm side in the real run.** The red-pen config is `pen_lora_right.yaml` but `configs/skills.yaml`
  calls the red-pen arm "LEFT main arm", so the hero shows no LEFT/RIGHT tag. The twin shot follows the
  twin's own convention (left opens, right places).
- **Speed.** Hero footage is sped up 4x then 2x, and the HUD says so. The 14-29 s approach is cut.
- **Text limit.** Captions stay within 6 words. Book pages, diagram labels, the task-tree panel and the
  terminal are interface or diegetic text and go over that on purpose.
- **Result and team** are empty in `config.json`; filled values appear on the end card.
- The repository has no licence file, so the video does not say "open source" about Liminal.
