# Public human / handheld / egocentric bimanual datasets for an EE50 fine-tune of FLUX 3 Action

Research date: Sat 17 Oct 2026. Nothing was downloaded; only dataset cards, READMEs, directory listings, HF API metadata (file trees, `meta/info.json`) and paper abstracts were read.
Every number has a source URL in the per-dataset notes. Anything not confirmed by a primary source (paper, official repo, official dataset card or file listing) is marked **UNVERIFIED**. Numbers marked "(est.)" are my own arithmetic from listed file sizes.

Target for the team: per-arm 6-DoF EE pose + gripper opening at 30 Hz, bimanual, tabletop pick-and-place / obstacle moving, for two SO-101 5-DoF arms with fin-ray grippers, a top-down head camera and two wrist cameras. EE50 per arm = 3 translation + 6 rotation (first two rows of R) + 16 hand/gripper dims (gripper 0 = closed, 1 = open), poses relative to an anchor frame.

---

## 1. What Black Forest Labs says about the midtraining data

Sources read: research page https://bfl.ai/models/flux-3-action (the page BFL cites as the tech report), HF blog https://huggingface.co/blog/black-forest-labs/flux-3-action, model card https://huggingface.co/black-forest-labs/flux-3-action-base, docs https://docs.bfl.ai/flux_3/flux3_action_overview, and the local clone of github.com/black-forest-labs/flux-action (README, docs/, configs/).

- **No midtraining dataset is named anywhere.** The research page gives only a category mix (Table 2): joint video/audio 36.95% of samples, gaming 19.55%, egocentric human hands 13.54%, handheld grippers 14.03%, teleop across 14 embodiments 15.93%. Only quote used in this document: "13.54% from egocentric videos with hand pose annotations, 14.03% from handheld grippers" (bfl.ai/models/flux-3-action).
- A keyword sweep of the research page found no mention of Aria, Vision Pro, ARKit, GoPro, UMI, MANO, fisheye, EgoDex, Ego4D, HOT3D, AgiBot, Open X or ALOHA. The only dataset named for training is DROID, and only for the post-midtraining fine-tune (plus a small in-house SO-101 teleop set).
- The 36-entry bibliography cites none of the datasets below. It does cite EgoBridge (Punamiya et al. 2025, arXiv 2509.19626) but only for the per-embodiment, anchor-offset-indexed z-score normalisation. EgoBridge's line of work (EgoMimic, EgoVerse, Georgia Tech) uses Aria-glasses human data, so this is at most a weak hint that BFL works with that style of data. It is not evidence that EgoMimic/EgoVerse data was used.
- Other facts from the page: action rates across datasets range 5 to 30 Hz; robot datasets in a 14-D joint space (6 joints + gripper per arm, ALOHA-style) are kept in joint space; EE50 pose is relative to the anchor (last conditioning frame) while hand articulation and gripper stay absolute.
- The local flux-action repo only documents DROID, SO-101 community data (`configs/so101/community_corpus.json`: MolmoAct2 SO-100/101 manifest + `lerobot/community_dataset_v3`), games and a drone. Nothing about EE50 sources.

**Consequence:** "already in FLUX" can only be judged by plausibility. Plausible (large, popular, pose-labelled, released before mid-2026): EgoDex, HOT3D, ARCTIC/OakInk2/GigaHands/TACO (hand-object mocap), EgoMimic/EgoVerse, Ego-Exo4D; handheld: UMI public sets, FastUMI-100K, GenRobot 10Kh, possibly HiFi-UMI-2K (July 2026, may be too late); teleop: AgiBot World, OXE. Data recorded after the FLUX release (23 Sep 2026) cannot be in it: this is confirmed only for part of Pantheon DiverseUMI (see 3.1).

---

## 2. Summary table

Legend for "In FLUX?": **Confirmed** = BFL names it; **Plausible** = no evidence, but popular and pre-dates FLUX; **Unlikely** = released too late or niche; **No** = recorded after release.

| Dataset | Type | Bimanual | Size | fps | Cameras / viewpoint | Pose data | Format | Download size | Licence / access | Conversion to per-arm 6-DoF + gripper @30 Hz | In FLUX? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Pantheon DiverseUMI sample25 / sample100** | handheld UMI-style grippers | yes (all) | 25.0 h / 4,587 eps; 99.6 h / 12,219 eps | 30 | top exo cam (fisheye or 360, rectified) + 2 wrist cams 640x480 | per wrist: gripper-tip xyz (m) + rotvec, opening 0 to 1; visual SLAM scaled by IMU; gravity vector; validity flags | per-episode folder: 3 mp4 + actions.npz + labels.json | 69 GB / 251 GB; per-episode folders, ~15 MB/ep (est.) | Pantheon Commercial Use Licence v1.0; HF gate **manual** approval | Easy: already 30 Hz, opening 0 to 1 matches EE50 gripper; only check whether L/R share a world frame | No for recordings after 2026-09-28 (100h sample); 25h sample UNVERIFIED |
| **FastUMI-100K** (IPEC-COMMUNITY/FastUMI_100k_lerobot) | handheld grippers (T265 VIO + fisheye) | 20 dual-arm tasks + single-arm | 100K+ traj, ~600 h, 54 tasks, 1.4 TB | 20 (from 60 Hz video) | wrist fisheye cams, 720x1280, 1 per arm | 7-D per arm: xyz + roll/pitch/yaw + gripper 0 to 1, in a relative EE frame (undefined) | LeRobot v2.1, per-episode mp4/parquet | ~19 MB/episode for 2 cams (est.) | Apache-2.0 (umi-data portal flags TODO); open, no gate | Medium: resample 20 to 30 Hz, Euler to rot6d, clarify the relative frame and L/R alignment | Plausible |
| **HiFi-UMI-2K** (simple-world-lab) | handheld grippers + head stereo rig | yes | 2,000 h released (of 20k+ h) | 25 | head stereo pair + 2 cams per hand (~200 deg), all 512x640 | per hand: xyz + rot6d (first two rows of R, same as EE50) + gripper angle (rad); shared gravity-aligned world frame | LeRobot v3.0 sharded `chunk-XXXX/part-0000` | 398 chunks; chunk-0000 = 40.5 GB (1,125 eps, 5.3 h); one camera file 6.5 to 7.2 GB | CC BY 4.0; open | Easy for pose (rot6d native, shared frame); 25 to 30 Hz resample; angle to 0..1 | Plausible but late (Jul 2026) |
| **GenRobot 10Kh-RealOmin-OpenData** | handheld DAS Gripper | yes | uploaded 1,603 h / 76,028 clips (5.37 TB) | 30 | 1 fisheye cam per gripper, 1600x1296 | `/robotN/vio/eef_pose` (foxglove Pose + vel); gripper width 0 to 0.103 m (magnetic encoder) | MCAP + protobuf, one file per episode | ~70 MB/clip (est.); repo total 36.9 TB | CC-BY-SA-4.0; HF gate **auto** (form) | Medium: need das-datakit to decode MCAP, frame docs external | Plausible |
| **IO-AI SenseXperience UMI** (ICRA-WBCD) | wearable rig with handheld grippers | partly (some tasks) | 1,051 eps, 220,963 frames (~2.0 h), 92 tasks | 30 | 2 head (2048x1536) + 2 wrist (2048x1536) + head depth + gripper cam | head/left/right gripper 7-D pose (xyz + quat); per-hand rad and dist state fields | LeRobot v3.0, ~200 MB video files | 26.3 GB total | Apache-2.0; open | Medium: frame, units and tracking method undocumented | Unlikely |
| **UMI** (Chi et al. 2024) public zarrs | handheld GoPro grippers | cloth folding and dish washing yes; cup/tossing no | cup wild 1,447 eps; cup lab 305; tossing 284; cloth 249; dish 258 | ~60 (GoPro); LeRobot copy says 10 (UNVERIFIED) | GoPro Hero 9 + 155 deg fisheye on each gripper | `robotN_eef_pos`, `robotN_eef_rot_axis_angle`, `robotN_gripper_width` (fiducial tags), ORB-SLAM3 mono-inertial | zarr.zip (+ raw mp4 zips) | cup wild 16 GB, cup lab 4.2 GB, cloth 14 GB, dish 21 GB; LeRobot cup wild 3.15 GB | MIT; open | Easy: native EE pose + width; downsample 60 to 30 Hz; 224x224 images only in zarr | Plausible |
| **EgoDex** (Apple) | human hands, Apple Vision Pro | yes (two hands always tracked) | 829 h, 338k demos (UNVERIFIED count), 194 tasks | 30 | egocentric 1080p (synthesised from AVP cameras) | ARKit SE(3) 4x4 for camera + ~68 joints (wrists, every finger joint), ARKit world frame, confidences | mp4 + hdf5; LeRobot v3 mirror per task | 5 x 300 GB train, 16 GB test, 200 GB extra; mirror test/basic_pick_place = 0.56 GB | CC-BY-NC-ND; open | Medium: wrist SE(3) direct; gripper proxy from thumb-index tip distance; head-to-table frame change | Plausible (strong candidate) |
| **PH2D** (Humanoid Policy ~ Human Policy) | human hands, Vision Pro / Quest 3 + ZED Mini | some tasks (pouring, cup passing) | ~27k human demos (UNVERIFIED), 41 task folders | UNVERIFIED | egocentric (AVP camera or ZED Mini) | head 6D rot, both wrists xyz + 6D rot, 10 fingertips | HDF5 per task folder | 16.2 GB total | MIT; open | Medium: wrist pose direct, gripper proxy from fingertips | Plausible |
| **EgoMimic** (gatech/EgoMimic) | human Aria + robot teleop | groceries and laundry yes | human: 1,400 / 160 / 590 demos (60 / 80 / 100 min) | 30 human, 50 robot | Aria glasses (human), robot cams | hand SE(3) from Aria MPS in current camera frame; no grasp signal | robomimic HDF5 | 6 files, 23 to 46 GB each (243 GB) | no data licence stated; open | Hard for small subset (monolithic 40 GB files); no gripper | Plausible (weak EgoBridge link) |
| **HOT3D** | human hands, Aria + Quest 3 | yes | 833 min, 3.7M+ images, 19 subjects, 33 objects | ~30 (inferred) | egocentric multi-view mono/RGB | MANO and UmeTrack hand poses, 6-DoF object poses, headset poses (mocap GT) | VRS + toolkit; HOT3D-Clips webdataset tars | Clips 762 GB on HF; full size UNVERIFIED | hand annos CC BY-NC-SA, rest CC BY-SA; sign-up form for full set | Medium: MANO wrist to 6-DoF easy, grasp proxy needed; short clips | Plausible |
| **Ego-Exo4D** hand pose | human, Aria + GoPro exo | yes (skilled activities) | 1,286 h total; hand-pose subset size UNVERIFIED | UNVERIFIED for annotations | Aria ego + 4 exo GoPros | 21 kp/hand, 2D (manual/auto) + 3D global keypoints at annotated timestamps | JSON per take | recommended set ~14 TiB; annotations downloadable alone (size UNVERIFIED) | licence agreement, ~48 h approval, 14-day AWS keys | Hard: keypoints only (no wrist rotation), sparse, not tabletop pick-place | Plausible |
| **Ego4D** | human, head-mounted | n/a | 3,670 h (secondary source) | n/a | head-mounted | no 3D hand pose (UNVERIFIED; only 2D hand/object boxes in FHO benchmarks) | | | licence agreement | Not usable for EE poses | Plausible (video only) |
| **HOI4D** | human, helmet Kinect v2 + RealSense D455 | mostly single (right) hand (UNVERIFIED) | 2.4M RGB-D frames, 4,000+ seqs | UNVERIFIED | egocentric RGB-D | MANO (3 global rot + 45 pose + 10 shape) | images + pkl/json (OneDrive/Baidu) | UNVERIFIED | CC BY-NC 4.0; open links | Medium, but not bimanual | Possible |
| **ARCTIC** | human, Vicon mocap | yes | 2.1M frames (fps UNVERIFIED) | UNVERIFIED | 8 allocentric + 1 egocentric | MANO + SMPL-X, ego camera trajectory, articulated object poses | npy/pkl + images | raw_seqs (all poses) 215 MB; cropped images 116 GB; full 649 GB | non-commercial; account registration (ARCTIC + MANO + SMPL-X) | Easy for poses; articulated-object tasks, not pick-place | Plausible |
| **TACO** | human, NOKOV mocap | yes (tool + target) | 2.5K seqs, 5.2M frames, 14 subjects | 30 | 12 allocentric 4096x3000 + 1 helmet L515 1920x1080 | MANO meshes, object poses | OneDrive/Dropbox | UNVERIFIED | dataset licence not stated | Medium; tool-use, not pick-place | Possible |
| **OAKINK2** | human, mocap | yes | 627 seqs / 4.01M frames (UNVERIFIED) | UNVERIFIED | allocentric + egocentric | MANO + SMPL-X, object 4x4 per frame | per-seq image tar + anno pkl on HF | 2.13 TB; per-seq anno 35 to 95 MB, image tar 0.6 to 20.6 GB | CC-BY-SA-4.0 (HF tag); open | Medium; complex multi-step tasks | Plausible |
| **GigaHands** | human, markerless 51-camera studio | yes | 34 h, 14k clips, 183M frames, 56 subjects, 417 objects | UNVERIFIED | 51 third-person views | 3D keypoints, MANO params, object poses | Globus tarballs | UNVERIFIED | CC BY-NC 4.0; Globus | Medium for poses; no ego/wrist view | Plausible |
| **HoloAssist** | human, mixed-reality headset (HoloLens 2, UNVERIFIED) | yes | 166 h (paper) / 169 h (site), 350 pairs | UNVERIFIED | egocentric RGB + depth | hand pose stream, head pose, gaze | per-modality archives | hand pose 219 GB; videos 145 to 184 GB | CDLA-Permissive-2.0; open | Hard: big, assembly-guidance tasks | Possible |
| **Gen-HumanEgo** (GenRobot, 2026) | human, 6-camera DAS-Ego rig | both hands tracked | 1,847.7 h, 44,632 eps, 10,257 tasks | 30 | 6 ego cams 1600x1300 + ego depth | 21 kp/hand, MANO, mesh | MCAP per episode | 62 TB | CC-BY-SA-4.0; auto gate | Medium-hard; huge, no gripper | Plausible (Sep 2026, borderline) |
| **EgoVerse** (2026) | human, stereo head rigs / iPhone | yes | 1,362 h, 80k eps (paper); site says 4,003 h | 30 (secondary) | head-mounted stereo fisheye | 21 kp/hand in camera frame + head SLAM (secondary) | via EgoDB / GitHub GaTech-RL2/EgoVerse | UNVERIFIED | UNVERIFIED | Medium-hard | Plausible |
| **AgiBot World Alpha** | robot teleop (not human) | yes | 92,214 traj / ~595 h | 30 | head + wrist cams (count UNVERIFIED) | flange pose per arm (xyz + quat xyzw), gripper mm / 0..1 | h5 + mp4 per task | 9.66 TB (8.5T); sample tar ~7 GB | CC BY-NC-SA 4.0; gated (contact info) | Easy for poses, but teleop robot data | Plausible |
| **OXE: utokyo_xarm_bimanual** | robot teleop | yes | 70 eps | UNVERIFIED | 1 cam 256x256 | 14-D: EE xyz + yaw/pitch/roll + gripper per arm | RLDS | 138 MiB | licence not stated | Easy but towel folding, tiny | Plausible |
| **YUBI / UMI Arena** (AIRoA) | handheld finger-aligned grippers + Quest tracking | yes | 579.9 h, 336,543 eps (release); 8,434 h in paper | 30 | 3 RGB 480x640 | quest/hand/fingertip poses xyz+quat in table frame; gripper angle | LeRobot v3.0 | 2.68 TB | NDA, competition only; manual gate | n/a (not usable) | Unlikely |

---

## 3. Per-dataset notes (with sources)

### 3.1 Pantheon DiverseUMI (sample25, sample100), new 2026 handheld bimanual data
- URLs: https://huggingface.co/datasets/Pantheon-inc/DiverseUMI-sample25 , https://huggingface.co/datasets/Pantheon-inc/DiverseUMI-sample100 , API https://huggingface.co/api/datasets?search=DiverseUMI
- What: two handheld UMI grippers plus a top-mounted camera, free-form bimanual tabletop manipulation with everyday objects.
- Size (cards): sample25 = 25.0 h, 4,587 episodes from 293 recordings, 77 skills, largest skills pick and place 2,669, stack 320, hand over 237, stand up 224; median episode 7 s. sample100 = 99.6 h, 12,219 episodes from 1,134 recordings (a search snippet showed 99.9 h / 12,356 eps, so the card was updated).
- fps: all three videos 30 fps, frame-aligned.
- Cameras: `exo.mp4` top camera (fisheye or 360 dual-fisheye source, delivered as a rectified table view, 512x328 to 1362x876), `left.mp4` / `right.mp4` wrist cams 640x480. From 2026-09-28 the top camera was mounted lower (sample100 card).
- Pose: `state` (N, 14) = per wrist gripper-tip xyz (m), rotation vector (rad), opening; actions (N-1, 7 per wrist) = relative motion in each wrist's own frame. Poses from visual SLAM on the wrist camera, scaled to metric with the gripper IMU, per-frame source codes and validity flags; per-frame gravity direction given. Each tracking chunk has its own (not gravity-aligned) world frame. Valid share (sample25): translation 96.2%, rotation 100%, opening 98.5%.
- Gripper: opening from wrist camera, normalised 0 = closed, 1 = open (matches EE50 gripper convention).
- Format: `freeform/<episode_id>/` with exo/left/right mp4, `actions.npz`, `labels.json`, `meta.json`; `manifest.jsonl` lists all episodes, so a subset can be selected with `allow_patterns`.
- Download: sample25 69 GB, sample100 251 GB. 69 GB / 4,587 eps = ~15 MB per episode (est.), so ~300 pick-and-place episodes is ~4.5 GB (est.).
- Licence/access: Pantheon Commercial Use License v1.0 (training and commercial model use allowed, no redistribution or competing datasets, attribution when sharing data). HF API reports `gated: "manual"`, i.e. the owner must approve the request. Faces are not blurred.
- Conversion: easiest of all candidates. Already 30 Hz, per-arm 6-DoF + opening 0..1, relative actions per wrist. Open question: whether left and right wrists share one SLAM world frame inside a chunk (the card says each tracking chunk has its own frame, it does not say per wrist). UNVERIFIED.
- In FLUX? The sample100 card documents recordings from 2026-09-28 onward, after the FLUX 3 Action release (2026-09-23 per the-decoder / MarkTechPost coverage), so at least those cannot be in midtraining. Both repos were last modified 2026-10-16. Whether the 25h sample pre-dates FLUX training is UNVERIFIED.

### 3.2 FastUMI-100K (and original FastUMI)
- URLs: https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot , paper https://arxiv.org/html/2510.08022 , dual-arm tree https://huggingface.co/api/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot/tree/main/dual_arm , meta https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot/raw/main/dual_arm/Take_Bottle_and_Place_on_Coaster/meta/info.json
- What: handheld UMI-style grippers with GoPro fisheye wrist views and a RealSense T265 for trajectory (VIO instead of offline SLAM) (paper).
- Size: 100K+ trajectories, ~600 h, 54 tasks; 1.4 TB on HF. 20 dual-arm tasks, e.g. Take_Bottle_and_Place_on_Coaster 2,227 eps / 910,682 frames (12.6 h, est.), Clean_Desktop 2,147 eps / 670,398 frames (9.3 h, est.), Dispose_of_Desktop_Debris 2,864 eps, Arrange_Toothbrush_and_Toothpaste 3,952 eps (demo counts per umi-data portal https://raw.githubusercontent.com/umi-data/umi-data.github.io/main/index.html).
- fps: 20 (video captured at 60 Hz, downsampled to 20 Hz per paper; `info.json` fps 20).
- Cameras: dual-arm: `left_camera_rgb_image` and `right_camera_rgb_image`, 720x1280, mpeg4. No head camera.
- Pose: 14-D state/action = per arm x, y, z, roll, pitch, yaw, gripper (0..1). Card says states/actions are in a relative end-effector frame without saying relative to what; its index ranges (0:6, 7:13) are inconsistent with 7 values per arm.
- Format: LeRobot v2.1, one parquet + one mp4 per camera per episode. Left-camera mp4s in Take_Bottle_and_Place_on_Coaster are 7.2 to 13.3 MB, ~9.5 MB average (from the API tree), so ~19 MB per bimanual episode (est.), ~250 episodes in 5 GB (est.).
- Licence/access: Apache-2.0 on the card (the umi-data portal source carries a TODO to confirm the licence); not gated.
- Original FastUMI (2024): https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI-Data (HDF5, MIT, auto-gated), paper https://arxiv.org/pdf/2409.19499 reports 10,000 demos over 22 tasks while PMLR page https://proceedings.mlr.press/v305/zhaxizhuoma25a.html says 15,000+ demos over 24 tasks; single-arm.
- Conversion: resample 20 to 30 Hz, Euler (convention undocumented, UNVERIFIED) to rot6d, establish whether both arms share one frame. T265 VIO drift is a known weakness.
- In FLUX? Plausible: released Oct 2025, very popular (93,757 downloads last month per HF API). No evidence.

### 3.3 HiFi-UMI-2K (Simple AI)
- URLs: https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K , card raw https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K/raw/main/README.md , paper https://arxiv.org/abs/2607.25895 , chunk info https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K/raw/main/chunk-0000/part-0000/meta/info.json
- What: robot-free bimanual UMI capture with head-mounted offline stereo-inertial SLAM, native relative pose between the two grippers, ~3 mm EE accuracy (paper abstract). 2,000 h released out of a 20,000+ h corpus (card).
- Cameras: `head_main`, `head_main_stereo_right`, `left_hand_up/down`, `right_hand_up/down`, all 512x640 h264 at 25 fps (info.json).
- Pose: 20-D = [right 10 | left 10], each xyz (m) + rot6d (first two rows of R) + gripper opening angle (rad). Shared world frame with +Z up (gravity aligned), arbitrary origin. Validity masks.
- Chunk 0: 1,125 episodes, 478,810 frames (5.3 h at 25 fps, est.), 38 tasks, total 40.5 GB: data parquet 80 MB, each camera mp4 6.5 to 7.2 GB (API tree). 398 chunks total.
- Licence/access: CC BY 4.0, not gated (HF API `gated: false`), 135k downloads.
- Conversion: the closest to EE50 (same rot6d, shared gravity frame); 25 to 30 Hz resample; gripper angle to 0..1 needs a max-angle calibration. Smallest video-bearing subset is one chunk plus one camera (~7.3 GB), above the 5 GB target.
- In FLUX? Plausible only if BFL's midtraining data cut-off was after late July 2026. UNVERIFIED.

### 3.4 GenRobot 10Kh-RealOmin-OpenData
- URL: https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData ; org list https://huggingface.co/api/datasets?author=genrobot2025
- What: Gen DAS Gripper data (handheld or wearable, card does not say), dual-gripper household tasks; robot0 = left, robot1 = right.
- Size: uploaded stage 1 = 950 h / 39,761 clips / 3.45 TB, stage 2 = 653 h / 36,267 clips / 1.92 TB; headline 13,000+ h; card shows conflicting totals (36.9 TB vs 95 TB).
- Sensors: one fisheye camera per gripper, 1600x1296, 30 fps, H.264; 6-axis IMU; gripper width from magnetic encoder, 0 to 0.103 m.
- Pose: `/robot0/vio/eef_pose` (foxglove Pose with velocities); frame and whether both grippers share a frame are documented in an external manual (UNVERIFIED).
- Tasks include desktop_object_sorting, drawer_to_take_items, small_object_storage, carton_sorting_clutter (household, partly tabletop).
- Format: MCAP + protobuf, one file per episode, converter das-datakit on GitHub. ~70 MB per clip on average (5.37 TB / 76,028, est.); median clip 210 s.
- Licence/access: CC-BY-SA-4.0, HF gate `auto` (contact form, instant).
- In FLUX? Plausible (Apr 2026, 266k downloads/month). No evidence.

### 3.5 IO-AI SenseXperience UMI (ICRA-WBCD)
- URL: https://huggingface.co/datasets/ICRA-WBCD/IO-AI-SenseXperience-UMI ; tree https://huggingface.co/api/datasets/ICRA-WBCD/IO-AI-SenseXperience-UMI/tree/main?recursive=true
- What: wearable human demo rig with left/right handheld grippers, head stereo, wrist cams; no robot actions.
- Size: 1,051 episodes, 220,963 frames (~2.0 h at 30 fps, est.), 92 tasks (pick-and-place, shelf/container, household, bimanual).
- Cameras: head left/right 2048x1536, left/right wrist 2048x1536, head depth (RGB-encoded) 800x600, gripper cam 800x600, 30 fps.
- Pose: `observation.pose.head/left_gripper/right_gripper` each 7-D [x y z qx qy qz qw]; `observation.state` 4-D per-hand rotation (rad) and dist values. Frame, units and tracking method are not documented.
- Format: LeRobot v3.0; data parquet 23 MB; ~130 video files of ~200 MB, so a subset is possible (~3 GB for a few files of 3 cameras, est.).
- Licence: Apache-2.0, not gated. Last modified 2026-05-12.

### 3.6 UMI (Chi et al. 2024) and the UMI data portal
- URLs: code https://github.com/real-stanford/universal_manipulation_interface , paper https://arxiv.org/html/2402.10329 , data index https://real.stanford.edu/umi/data/ , portal https://umi-data.github.io/ (table in https://raw.githubusercontent.com/umi-data/umi-data.github.io/main/index.html)
- Released zarrs (portal + directory listings): cup in the wild 1,447 demos (16 GB, https://real.stanford.edu/umi/data/zarr_datasets/), cup in the lab 305 demos (4.2 GB zarr, 28 GB mp4s, https://real.stanford.edu/umi/data/cup_arrangement/), dynamic tossing 284 demos, **bimanual cloth folding 249 demos (14 GB zarr, 52 GB mp4s, https://real.stanford.edu/umi/data/bimanual_cloth_folding/)**, **bimanual dish washing 258 demos (21 GB zarr, 51 GB mp4s, https://real.stanford.edu/umi/data/dish_washing/)**. All MIT.
- Hardware: GoPro Hero 9 + 155 deg fisheye per gripper; poses from ORB-SLAM3 monocular-inertial; gripper width from fiducial markers on the fingers; policies run at 10 Hz (20 Hz tossing) (paper).
- Zarr keys: `robot{i}_eef_pos`, `robot{i}_eef_rot_axis_angle`, `robot{i}_gripper_width`, `robot{i}_demo_start_pose/_end_pose`, `camera{i}_rgb` at 224x224 (07_generate_replay_buffer.py). Index 0 = right, 1 = left; both grippers localised in one map (06_generate_dataset_plan.py). Timing comes from the GoPro stream rate (59.94 used in the script's test); the portal describes the example zarr as ~60 Hz.
- LeRobot copy https://huggingface.co/datasets/lerobot/umi_cup_in_the_wild: 1,447 eps, 699,432 frames, 3.15 GB, Apache-2.0, labelled 10 fps (inconsistent with ~60 Hz source, UNVERIFIED), single arm.
- Conversion: easy (native EE pose + width). Zarr images are only 224x224 fisheye; the bimanual zips are monolithic (14 to 21 GB), so no clean <5 GB bimanual subset.
- Related: UMI on Legs tossing session zarr ~1.4 GB (search result for https://real.stanford.edu/umi-on-legs/tossing/, single arm); DexUMI sample https://real.stanford.edu/dexumi/sample_data.zip (single dexterous hand, per https://github.com/real-stanford/DexUMI). Other bimanual entries in the portal: iPhUMI sweater folding (https://huggingface.co/datasets/austinpatel/iphumi_bimanual_laundry_folding, 1,572 demos, MIT), TAMEn (https://huggingface.co/datasets/OpenDriveLab-org/TAMEn, 713 demos, tactile, CC BY-NC-SA 4.0), HuMI (whole-body, CC BY 4.0), DexWild (https://huggingface.co/datasets/boardd/dexwild-dataset, dexterous hands).

### 3.7 Other UMI-style HF datasets seen (HF search for umi, https://huggingface.co/datasets?search=umi)
- UMI-Benchmark-v1 https://huggingface.co/datasets/UMIbenchmark/UMI-Benchmark-v1 : 20,000 sessions, 10 tasks (6 dual-arm: pour beans, pack and carry, turntable pick, mahjong sort, kitchen rearrange, pants folding), 2.41 TB tar shards (~16 GB each), CC-BY-4.0, hardware/fps/pose not documented.
- YUBI UMI Arena https://huggingface.co/datasets/airoa-org/yubi-corl2026-umi-arena : 579.9 h, 336,543 eps, 30 Hz, 3 RGB 480x640, poses in a table-calibrated frame, 2.68 TB, NDA + competition-only. Paper https://arxiv.org/abs/2606.10244 (8,434 h, 1.20M eps).
- PrimeBot challenge https://huggingface.co/datasets/challenge-2026/challenge_data : PrimeUMI shirt folding, two 960x960 wrist cams at 30 fps, per hand xyz + quat (wxyz) + gripper angle in degrees, hands in independent frames; UMI sample 5 eps = 0.48 GB; repo 5.24 TB; CC-BY-SA-4.0.
- brandonyang/dual-lidar-umi https://huggingface.co/datasets/brandonyang/dual-lidar-umi : 115 eps, 110,269 frames, 30 fps, 2 cams 800x600, per-UMI xyz + rx/ry/rz, gripper width (mm), 35.2 GB, Apache-2.0, no task description. brandonyang/umi-fold-shirt: 27 eps, 30 fps, 9.43 GB.
- TeleEmbodied/VISTA-UMI-5K https://huggingface.co/datasets/TeleEmbodied/VISTA-UMI-5K : 30 tasks, apache-2.0, two-arm action fields, ~20 fps inferred; sparse card.
- zhb10086/umi-toybench-dual: synthetic PyBullet, private. open-index/umi-robots: a robots.txt web crawl, unrelated.
- RDT2 (https://raw.githubusercontent.com/thu-ml/RDT2/main/README.md): 10,000+ h of UMI-style data, not released as far as found; only an example set robotics-diffusion-transformer/BimanualUR5eExample.
- MV-UMI (https://arxiv.org/abs/2509.18757): no official dataset release found; ActiveUMI (https://arxiv.org/abs/2510.01607): VR kit, no dataset release found.

### 3.8 EgoDex (Apple)
- URLs: https://github.com/apple/ml-egodex , paper https://arxiv.org/abs/2505.11709 , LeRobot mirror https://huggingface.co/datasets/griffinlabs/EgoDex-LeRobot-v3.0
- What: Apple Vision Pro egocentric video with ARKit hand tracking, 194 tabletop tasks, 829 h (725 h train, 7 h test, 97 h extra), 1080p, 30 Hz (README). 338,000 demos / 90M frames comes from the paper text as quoted by alphaxiv (https://www.alphaxiv.org/abs/2505.11709), not re-checked in the primary PDF (UNVERIFIED); 829 h x 30 Hz = 89.5M frames is consistent.
- Pose: per-frame 4x4 SE(3) for the camera and ~68 joints (head, arms, `leftHand`/`rightHand` wrists, all finger joints) in the ARKit origin frame of each session, optional confidences; LLM-written task descriptions.
- Download: train parts 5 x 300 GB, test 16 GB, extra 200 GB (README). The LeRobot mirror splits by task: `test/basic_pick_place` = 277 eps, 32,271 frames, 0.56 GB, AV1 1080p (tree + info.json). Not gated.
- Licence: CC-BY-NC-ND (README); the mirror keeps cc-by-nc-nd-4.0. The no-derivatives clause matters if you redistribute converted data.
- Conversion: wrist 6-DoF directly from `leftHand`/`rightHand`; gripper proxy from thumb-tip to index-tip distance; express relative to the camera or a gravity-aligned frame; filter low-confidence frames. Human wrist motion is far outside an SO-101 workspace.
- In FLUX? Plausible and likely: the biggest open egocentric 3D-hand dataset; many WAM pretraining mirrors exist on HF (e.g. dreamzero-egodex-pretrain repos). No direct evidence.

### 3.9 PH2D
- URLs: https://huggingface.co/datasets/RogerQi/PH2D , project https://human-as-robot.github.io/ , paper https://arxiv.org/html/2503.13441v3
- What: egocentric human demos with Apple Vision Pro (ARKit) or Quest 3 / AVP + ZED Mini; state includes 6D head rotation, both wrists xyz + 6D rotation, 10 fingertip positions (paper). ~27k human demos (dozens of hours) per project page and figure caption; frame count 3.02M only from a secondary review (UNVERIFIED).
- Tasks (task folders in the HF tree): grasping (cans, bottles, cubes), picking onto colour pads, pouring (bimanual), cup passing (bimanual), mostly tabletop.
- Download: 16.2 GB total, HDF5 per task folder, MIT, not gated. fps UNVERIFIED.
- Conversion: like EgoDex (wrist pose direct, fingertip distance as gripper proxy).

### 3.10 EgoMimic
- URLs: https://github.com/SimarKareer/EgoMimic , https://huggingface.co/datasets/gatech/EgoMimic , paper https://arxiv.org/html/2410.24221
- Human data from Aria glasses at 30 Hz; hand SE(3) in the current camera frame from Aria MPS; no grasp signal (the paper notes Aria records hand pose only). Human data: object-in-bowl 1,400 demos / 60 min, groceries 160 / 80 min (bimanual), laundry 590 / 100 min (bimanual). Robot data is ALOHA-style at 50 Hz.
- Files: 6 HDF5 files 23 to 46 GB each, 243 GB total; no dataset licence stated (code MIT).

### 3.11 HOT3D
- URLs: https://github.com/facebookresearch/hot3d , paper https://arxiv.org/abs/2411.19167 , HOT3D-Clips https://huggingface.co/datasets/bop-benchmark/hot3d , licence https://projectaria.com/datasets/hot3D/license
- 833+ min, 3.7M+ images, 19 subjects, 33 objects, Aria + Quest 3; UmeTrack and MANO hands, mocap object poses, headset poses. Clips: 762 GB webdataset, ~30 Hz from timestamps (inferred). Full set via sign-up and 14-day links. Licence: hand annotations CC BY-NC-SA, recordings and other annotations CC BY-SA (per licence page summary).

### 3.12 Ego-Exo4D and Ego4D
- URLs: https://docs.ego-exo4d-data.org/annotations/ego_pose/ , https://docs.ego-exo4d-data.org/getting-started/ , paper https://arxiv.org/abs/2311.18259
- 1,286 h, 740 participants, 123 scenes. Hand pose: 21 keypoints per hand, 2D on ego/exo cams (manual or auto) and global 3D keypoints at annotated timestamps; JSON per take; downloadable alone with `--parts annotations --benchmarks handpose`. Access: licence agreement, ~48 h approval, AWS keys valid 14 days; the recommended download set is ~14 TiB.
- Ego4D: no 3D hand-pose annotations found (only 2D hand/object boxes in its Hands and Objects benchmarks), UNVERIFIED in this session.

### 3.13 HOI4D, ARCTIC, TACO, OAKINK2, GigaHands, HoloAssist (lab hand-object datasets)
- HOI4D https://hoi4d.github.io/ , https://arxiv.org/abs/2203.01577 : 2.4M RGB-D egocentric frames, 4,000+ seqs, 800 object instances, 16 categories (abstract: 4 participants; CVPR version: 9); helmet with Kinect v2 + RealSense D455 and MANO parameters per the CVPR paper and annotation docs (via search results); right-hand only is UNVERIFIED; CC BY-NC 4.0; OneDrive/Baidu links.
- ARCTIC https://github.com/zc-alexfan/arctic , data README https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md : 2.1M frames, 8 allocentric + 1 egocentric views, 54-camera Vicon, MANO + SMPL-X; `raw_seqs` (MANO/SMPL-X params, ego camera trajectory, object poses) only 215 MB, cropped images 116 GB, full images 649 GB; registration with ARCTIC + MANO + SMPL-X accounts; non-commercial.
- TACO https://taco2024.github.io/ , https://arxiv.org/html/2401.08399 : 2.5K seqs, 5.2M frames, 12 allocentric 4096x3000 + 1 helmet L515 1920x1080, 30 Hz, 14 subjects, 131 tool-action-object triplets, MANO; NOKOV mocap; dataset licence not stated.
- OAKINK2 https://github.com/oakink/OakInk2 , https://huggingface.co/datasets/kelvin34501/OakInk-v2 : bimanual, allocentric + egocentric, MANO + SMPL-X, per-frame object 4x4; HF 2.13 TB, cc-by-sa-4.0 tag; per-sequence anno pkl 35 to 95 MB and image tar 0.6 to 20.6 GB (API tree). Sequence/frame counts (627 / 4.01M) UNVERIFIED.
- GigaHands https://github.com/Kristen-Z/GigaHands , https://arxiv.org/abs/2412.04244 : 34 h, 14k clips, 183M frames, 56 subjects, 417 objects, 51 views, markerless; keypoints + MANO params + object poses on Globus; CC BY-NC 4.0.
- HoloAssist https://holoassist.github.io/ , https://arxiv.org/abs/2309.17024 : 166 h (abstract) / 169 h (site), 350 instructor-performer pairs; hand pose 219 GB, head pose 4.7 GB, videos 145 to 184 GB; CDLA-Permissive-2.0.
- Common conversion note: MANO global orient + wrist translation gives a 6-DoF wrist pose directly; gripper opening must be derived (thumb-index distance); views are mostly third person, so a head-cam/wrist-cam observation pairing like our robot is missing (ARCTIC/TACO/OAKINK2 do have one egocentric view).

### 3.14 2026 large egocentric releases
- Gen-HumanEgo https://huggingface.co/datasets/genrobot2025/Gen-HumanEgo : 1,847.7 h, 44,632 eps, 10,257 tasks, 6-camera DAS-Ego rig 1600x1300 at 30 fps, 21 kp/hand + MANO + mesh, MCAP, 62 TB, CC-BY-SA-4.0, auto gate. Sample set Gen-EgoData https://huggingface.co/datasets/genrobot2025/Gen-EgoData : 4.23 h, 500 samples, 47.6 GB.
- EgoVerse https://arxiv.org/abs/2604.07607 , https://egoverse.ai/ : 1,362 h / 80k eps / 1,965 tasks / 2,087 demonstrators (paper); site shows 4,003 h; capture rigs, 30 fps and 21-kp hands come from secondary write-ups (UNVERIFIED); licence and access not stated.
- Open-AoE https://arxiv.org/abs/2607.14183 : ~2,000 h smartphone video, MANO hands, camera trajectories; download not found.
- OpenEgo https://arxiv.org/abs/2509.05513 : 1,107 h unified from six public datasets; release site openegocentric.com.
- EgoScale https://arxiv.org/abs/2602.16710 : trains on 20,854 h of egocentric data, no new dataset release found.
- Build AI Egocentric-10K/100K/1M (https://huggingface.co/datasets/builddotai/Egocentric-10K): video only, per a secondary blog (https://www.labellerr.com/blog/egocentric-datasets-robotics/); no hand poses (UNVERIFIED).

### 3.15 AgiBot World and Open X-Embodiment (only bimanual EE-pose content)
- AgiBot World Alpha https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha , https://github.com/OpenDriveLab/AgiBot-World : teleop dual-arm; `state/end/position` [N,2,3] and `orientation` [N,2,4] (xyzw, flange) per arm; gripper in mm (state) and 0..1 (action, 1 = closed); 30 Hz; Alpha 92,214 traj (~8.5T per GitHub, 9.66 TB per HF), ~595 h after update; sample tar ~7 GB; CC BY-NC-SA 4.0; gated with contact info.
- OXE utokyo_xarm_bimanual https://www.tensorflow.org/datasets/catalog/utokyo_xarm_bimanual_converted_externally_to_rlds : 70 eps, 138 MiB, towel folding, 14-D EE xyz + yaw/pitch/roll + gripper per arm, one 256x256 camera. ALOHA-family OXE data is joint space (and BFL says it keeps 14-D joint datasets in joint space).

---

## 4. Ranking: small (<5 GB subset), downloadable tonight, bimanual, 6-DoF gripper pose + opening, tabletop pick-and-place

1. **FastUMI-100K dual_arm subset** (IPEC-COMMUNITY/FastUMI_100k_lerobot). Open, no gate, per-episode LeRobot v2.1 files, so ~200 to 250 episodes of `Take_Bottle_and_Place_on_Coaster`, `Clean_Desktop` or `Dispose_of_Desktop_Debris` fit in ~4 to 5 GB (est.). Per-arm xyz + RPY + gripper 0..1. Minuses: 20 Hz, Euler, ambiguous relative frame, wrist cams only, probably already in FLUX's handheld mix.
2. **Pantheon DiverseUMI sample25 subset**. Best content match (bimanual tabletop pick-and-place, stack, hand over; top cam + 2 wrist cams like our rig; 30 Hz; opening 0..1; at least partly recorded after FLUX was released, so new to FLUX). Select ~300 pick-and-place episodes via `manifest.jsonl` (~4.5 GB est.). Blocker: HF gate is manual approval, so request access immediately; if not approved tonight, it drops out.
3. **EgoDex per-task subset** (griffinlabs/EgoDex-LeRobot-v3.0, e.g. `test/basic_pick_place` 0.56 GB plus a few more tabletop tasks). Instant, open, 30 Hz, bimanual human wrists. Minuses: no gripper (thumb-index proxy), CC-BY-NC-ND, likely already in FLUX's egocentric mix, so it adds little.
4. **IO-AI SenseXperience UMI** (ICRA-WBCD). Open, Apache-2.0, 30 Hz, head + both wrist cams (same layout as our robot), 7-D poses for both grippers, ~200 MB file granularity so a ~3 GB slice is possible. Minuses: frame, units and tracking undocumented; only part of the 92 tasks are bimanual.
5. **GenRobot 10Kh-RealOmin-OpenData subset** (e.g. `desktop_object_sorting`, `small_object_storage`). Auto gate (instant), bimanual, 30 Hz fisheye wrist cams, gripper width in metres, one MCAP per episode (~70 MB, so ~60 clips in 4 GB, est.). Minuses: MCAP/protobuf needs GenRobot's das-datakit, pose frame documented externally, probably already in FLUX.

Honourable mentions: **HiFi-UMI-2K** (best pose quality, rot6d and shared gravity frame exactly like EE50, CC BY 4.0, open), but the smallest useful slice is ~7.3 GB (one chunk, one camera). **PH2D** (MIT, 16.2 GB, tabletop grasp/pour, wrist 6-DoF, per-task folders) if a human-hand source is wanted. **UMI cup_in_the_lab** zarr (4.2 GB, MIT) is single-arm; UMI bimanual cloth/dish zarrs are 14/21 GB.

General conversion caveats for SO-101: the arms are 5-DoF, so human/handheld 6-DoF orientations must be projected to the reachable set (or filtered); handheld workspaces are much larger than SO-101 reach; fin-ray width range differs from UMI-style grippers (normalise to 0..1); wrist-camera optics differ (fisheye vs our wrist cams). For EE50, relative-to-anchor poses reduce, but do not remove, the need for a consistent TCP definition and a gravity-aligned frame.
