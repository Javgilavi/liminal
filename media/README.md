# Media: ERH Physical AI Hackathon, 17 to 18 October 2026

Real material from the Haller rig (two SO-101 arms on a static XLeRobot tower), its MuJoCo digital twin
(`examples/dual_so101_sim/`), FLUX 3 Action runs and the liminal UIs. Every caption says what the clip really shows;
simulated runs are labelled as such, and no clip here shows a learned policy completing a task.

## `real_robot/`

| File | What it shows |
|---|---|
| `haller_tabletop_real_scene_2026-10-18.mp4` | The real Haller tabletop (tower, two SO-101 arms, D455, round table, mat, box, markers, cutter). The reference the digital twin was measured from. |

## `digital_twin/`

| Folder / file | What it shows |
|---|---|
| `rig/dual_so101_rig_{front,top,overhead,left_wrist,right_wrist}.png` | The bare dual SO-101 rig (`scene.xml`) from every camera: `top` is the D455 on the tower, wrist cameras at the lens face. |
| `tabletop/real_video_reference_frames_contact_sheet.jpg` | Frames from the real video used to place the table, mat, box and objects. |
| `tabletop/twin_tabletop_*.png`, `twin_tabletop_scene_preview.mp4` | The reconstructed tabletop twin from the front (lid closed and open), D455, overhead and both wrist cameras. |
| `tabletop/expert_grasp_inspection.jpg`, `expert_successful_demonstration.jpg` | Scripted expert diagnostics: the marker grasp, and a completed scripted place. |
| `expert_demos/expert_place_green_marker_*.mp4` | Scripted expert demonstrations used as LoRA training data (243 kept of 576 generated: green 133/288, red 110/288). |
| `lora_eval/*_zero_shot.mp4` vs `*_lora_step1600.mp4` vs `*_lora_final.mp4` | Same held-out seeds, FLUX zero-shot vs the twin LoRA. No checkpoint completes a place (0/20 each), but the LoRA brings the gripper to the marker: red seed 901001 ends 20.6 cm from it zero-shot and 2.2 cm with the final LoRA; green seed 900000 37.3 cm zero-shot, 5.3 cm at step 1600. |
| `lora_eval/twin_lora_training_loss.png` | Training loss of the twin LoRA. |

## `flux_sim/`

FLUX 3 Action zero-shot in MuJoCo: real inference on a GPU, simulated arms, no task-success scoring.

| File | What it shows |
|---|---|
| `flux_zero_shot_{pick_green_block,pick_blue_block,green_block_to_blue_box}_rollout.mp4` | The simulated rollout for each prompt. |
| `flux_zero_shot_*_observed_vs_imagined.mp4` | One action chunk: what the camera saw next to the future FLUX predicted ("imagined") for it. |

## `liminal_ui/`

| File | What it shows |
|---|---|
| `haller_web_landing_page.png` | The Haller workbench page: real setup photo, interactive twin, recorded demo and live-lab panel. |
| `orchestrator_demo_verification.{mp4,png}`, `orchestrator_detailed_view.png` | The liminal orchestrator demo and its verification view. |
| `live_ui_flux_mujoco_session.mp4`, `live_run_flux_mujoco_cameras.mp4` | A live liminal session with real FLUX inference driving simulated arms, and its camera reel (top + both wrists). |
| `live_ui_recorded_video_analysis_session.mp4`, `live_run_recorded_video_cameras.mp4` | The same UI on a recorded human assembly video: observational VLM analysis, no robot commands. |
| `pen_replacement_controller*.{png,mp4}`, `pen_replacement_maintenance_plan.png` | The minimal five-step pen-replacement controller page (photo-review mode, no robot connected) and its arm/checkpoint plan. |

Sources: `~/erh_hack/` (real video), `~/erh_hack/dual_so101_sim/` (`renders/`, `reconstruction/`, `twin_results/`,
`flux/runs/`) and `liminal/runs/`, `liminal/outputs/` on the hackathon laptop.
