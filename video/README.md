# Liminal launch video

A 30-second product-style showcase, built in code with [Remotion](https://www.remotion.dev/).
Renders: [`media/launch_video/liminal_launch_16x9.mp4`](../media/launch_video/liminal_launch_16x9.mp4)
(1920x1080) and [`liminal_launch_9x16.mp4`](../media/launch_video/liminal_launch_9x16.mp4) (1080x1920),
30 fps, H.264 + AAC.

Every frame is either repo footage under `../media` or a motion graphic stating a fact from the repo.
Simulated footage always carries the filled `SIM` tag; real footage carries `REAL`; FLUX's predicted
video is tagged `IMAGINED · FLUX`.

## Edit and re-render

```bash
cd video
npm ci
python3 scripts/prepare_assets.py   # crops/upscales ../media into public/, tracks the pen, writes music.wav
npm run studio                       # live preview
bash scripts/render.sh               # renders both formats into ../media/launch_video/
```

All on-screen text, colours, the grade, shot timings, hero speed ramps and diagram layout live in
[`config.json`](config.json). Keep cuts on the 0.5 s beat grid. Set `"music": ""` to render silent,
or drop any 120 BPM track in `public/` and point `music` at it.

| File | Role |
|---|---|
| `config.json` | Text, colours, grade, timings, diagram nodes and positions |
| `scripts/prepare_assets.py` | Builds `public/` from `../media` (crops, 3x Lanczos upscale, still frames) |
| `scripts/track_red_pen.py` | Finds the red pen cap per frame in the wrist camera; the HUD box follows it |
| `scripts/generate_music.py` | Synthesizes the original 120 BPM score (no licence needed) |
| `scripts/stills.mjs` | Renders review stills: `node scripts/stills.mjs Liminal16x9 /tmp/out 7.5 14` |
| `src/scenes/*.tsx` | Opening (hook, problem), Hero, Middle (imagination, diagram, UI, twin), Closing |
| `src/components/ui.tsx` | Shared cards, tags, chips, HUD brackets, wordmark, transitions, grain |

## 1. Style reference

The LibreYOLO reference link was a placeholder, and no public LibreYOLO video turned up, so the
reference video was **not watched**. The style below is the brief's description of a premium dev-tool
launch, not a copy of LibreYOLO: near-black canvas, one mint accent (`#3EE3A3`, picked to sit with
the green rig), Inter for headlines with tight tracking, JetBrains Mono for HUD and code, floating
rounded cards with 3D tilt and deep shadows, expo-out easing everywhere, masked word reveals, whip and
zoom transitions with motion blur, and film grain plus a vignette over everything. Pacing is about one
cut every 1.5 to 3 s, on a 120 BPM grid. Send the real link and the timings and type can be retuned to it.

## 2. Project summary

- **What it is:** a dual-arm orchestration layer over FLUX 3 Action. One arm changes or holds
  something in the scene so the other arm can do what it otherwise couldn't.
- **Hardware:** two SO-101 arms on a static XLeRobot tower (the "Haller" rig), an Intel RealSense D455
  head camera and two wrist cameras.
- **Learned skills:** FLUX 3 Action runs bounded, text-prompted skills on one arm, adapted with task
  LoRA; dual-arm profiles run one independent FLUX instance per arm.
- **Coordination:** a deterministic behavior tree decides what runs, when, on which arm, and what must
  stay true; a safety supervisor gates every motor command; Qwen3-VL-4B perception checks that each
  goal predicate actually holds in the world state.
- **Evidence:** the real red-pen inserts with the red-pen LoRA (3 of 5 and 2 of 3 per block, as the
  team reports). Learned dual-arm performance on hardware is not verified, so the video shows
  dual-arm coordination only through the architecture and the UI.

## Asset notes

Tags: **REAL POV** robot cameras, **REAL RIG** photos and phone video of the rig, **SIM**, **UI**.

| Asset | Tag | Size, length | What it shows / best moments |
|---|---|---|---|
| `real_robot/flux_red_pen_runs/…layout_C_48s.mp4` | REAL POV | 640x240 (2x 320x240), 47.9 s | **Used.** Head + wrist cams. Grasp 2-14 s; pen lowered into slot 29-37 s; released and gripper withdraws 37-39 s; a hand resets the scene from 42 s. |
| `…layout_A_94s.mp4`, `…layout_B_91s.mp4` | REAL POV | 640x240, 94 / 91 s | Same task, longer approaches; C has the cleanest insert. |
| `…morning_A_69s.mp4`, `…morning_C_36s.mp4` | REAL POV | 640x240, 69 / 36 s | 07:40 block; morning C pen ends near but less visibly in the slot. |
| `real_robot/flux_imagination/…lora5k_ep101.mp4` | REAL POV + model | 1048x340, 9.9 s | **Used.** Actual vs imagined, held-out take 101; 4.6-6.1 s the wrist views match closely. |
| `real_robot/haller_gripper_closeup.jpg` | REAL RIG | 1512x2016 | **Used.** Macro of the gripper fingers. |
| `real_robot/haller_xlerobot_station.jpg` | REAL RIG | 1600x1200 | **Used.** The whole tower, both arms, HALLER label. |
| `real_robot/haller_tabletop_real_scene_2026-10-18.mp4` | REAL RIG | 478x850, 20.4 s, 120 fps | Handheld walk around the table; strong 0-5 s, too shaky later. |
| `real_robot/haller_rig_overview.jpg`, head-camera stills | REAL RIG / POV | 640x720, 640x480 | Smaller duplicates of the shots used. |
| `digital_twin/real_vs_sim/haller_photo_vs_sim.jpg` | REAL RIG + SIM | 1280x1440 grid | **Used** (top row): photo next to the MuJoCo view. |
| `digital_twin/lora_eval/*.mp4`, `expert_demos/*.mp4` | SIM | 480x360 / 660x360, 9-14 s | Twin LoRA evals (no completed place) and scripted demos. |
| `flux_sim/*.mp4`, `block_cup/*.gif`, `lidded_box/*.jpg` | SIM | 800x600 and smaller | Zero-shot FLUX in MuJoCo; closed-loop overlay is 117 s. |
| `liminal_ui/live_ui_flux_mujoco_session.mp4` | UI (SIM arms) | 1440x1000, 82 s, 1 fps | **Used** (frame at 22 s): reactive task tree with per-step predicates. |
| `liminal_ui/orchestrator_*`, `pen_replacement_*` | UI | 1440 wide | Static verification output and controller pages. |
| `analysis/*.png` | figures | | Kinematics plots; not footage. |
| Drive `IMG_6374/6390/6397.MOV`, `FB496A37….jpg` | REAL RIG | 28-164 MB | Listed, not downloaded: this session's network blocks Drive downloads. Its `00_highlights/` mirrors repo files. |

## 3. Storyboard (exact sources)

Beat grid: 120 BPM, one beat = 0.5 s, bar = 2 s. Every cut lands on a beat.

| Time | Shot | Source and timestamp | Tag | On screen |
|---|---|---|---|---|
| 0.0-1.5 | Hook macro, slow push | `haller_gripper_closeup.jpg` | REAL RIG | — |
| 1.5-3.0 | Hard cut, pull-out; wordmark masks in | `haller_xlerobot_station.jpg` | REAL RIG | LIMINAL · dual-arm · SO-101 · FLUX 3 Action |
| 3.0-4.5 | Kinetic type | — | — | One arm can't do everything. |
| 4.5-6.0 | Kinetic type, whip out | — | — | Two arms. Explicit dependencies. |
| 6.0-9.0 | Hero, head + wrist cards, 4x | layout C 2.0-14.0 s | REAL POV | prompt typed; tracking box on the red cap (from `track_red_pen.py`) |
| 9.0-14.5 | Hero, speed ramp to 2x | layout C 29.0-40.0 s | REAL POV | slot target from 33 s; `red_pen_in_slot` flips to ✓ at source 38.0 s (released); "3 of 5 · team-reported" from 11.5 |
| 14.5-16.0 | Imagined vs actual | `…lora5k_ep101.mp4` 4.6-6.1 s, wrist crops | model / REAL POV | Imagine, then act. |
| 16.0-19.5 | Architecture diagram draws on | `config.json` nodes | — | Goal → behavior tree → skill registry → FLUX left / right → safety → arms → cameras → Qwen3-VL → world state ✓ |
| 19.5-21.0 | Live UI card over dimmed diagram | `live_ui_flux_mujoco_session.mp4` @ 22 s | SIM (UI) | — |
| 21.0-23.0 | Real rig vs twin | `haller_photo_vs_sim.jpg` top row | REAL RIG + SIM | Twin measured from the real rig. |
| 23.0-26.0 | Terminal card | README quick start; output captured from a real `liminal run` | — | Runs on mock arms. No hardware. |
| 26.0-30.0 | End card, held | `haller_xlerobot_station.jpg`, blurred | — | LIMINAL · tagline · ERH Physical AI Hackathon · Oct 2026 · Imagine Then Act track · repo |

Music impacts: 1.5, 6.0, 16.0 and 26.0 s; risers lead into 6.0, 16.0 and 26.0; drums drop out for the
imagination shot (14.5-16.0) and the end card.

## Choices to know about

- **Arm side.** The red-pen config is `pen_lora_right.yaml` but `configs/skills.yaml` calls the
  red-pen arm "LEFT main arm", so the hero shows no LEFT/RIGHT tag.
- **Speed.** Hero footage is sped up 4x then 2x, and the HUD says so. The 14-29 s approach is cut.
- **Text limit.** Headlines stay within 6 words. The diagram node labels and terminal lines are
  interface, not captions, and go over that limit on purpose.
- **Result and team** are empty in `config.json`; filled values appear on the end card.
- The repository has no licence file, so the video does not say "open source".
