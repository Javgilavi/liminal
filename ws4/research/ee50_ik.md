# EE50 on XLeRobot's 5-DoF arms: the IK gate (Simone's plan, step 2)

Owner: ws4-ik. Code: [`ws4/kinematics/`](../kinematics/README.md). Numbers: `ws4/kinematics/results.json`
(regenerate with `python -I ws4/kinematics/validate.py`, about 3 minutes on 16 CPU cores). Written 2026-10-17.

## Status

**Done.** EE50 convention audit against the released code. SO-101 forward kinematics from the official URDF
(matches MuJoCo to 4 µm). EE50 codec (encode, decode, per-offset z-scores) with round-trip tests. Constrained
analytic IK plus a damped-least-squares reference. The gate itself: 25,168 real joint frames from 5 Hub datasets
(single SO-101, bimanual SO-101, XLeRobot), 1,500 human handheld (UMI) chunks, 1,500 synthetic 6-DoF chunks, a
policy-noise study, table-workspace maps, and mesh collisions on the ws4-sim MuJoCo model. 31 CPU tests pass
(MuJoCo and cache tests skip when those are absent) and ruff is clean.

**Verified here** (measured or read in code): the FK chain and joint limits (URDF lines cited below), LeRobot's
degrees-to-radians convention, the codec round trip, every metric in the tables below.

**Not verified.**
- The EE50 conventions that the release does not contain. These are the coordinate frame of the relative pose,
  translation units, which of the 16 auxiliary dims holds the gripper, how the z-scores are stored, the name of the
  EE50 head inside the gated base checkpoint, and the state input of EE50 embodiments. All are marked UNVERIFIED.
- The fin-ray TCP and the arm mount poses. Both come from CAD via `ws4/sim/station_geometry.yaml` and are not yet
  measured on the robot.
- The unit (degrees or M100) of each public dataset. It is guessed from value ranges.
- Nothing has run on hardware.

**Next 3 steps.**
1. ws4-flux-dual: read the safetensors header of `flux-3-action-base` (needs an HF token with the gate accepted) to
   get the EE50 head name and widths. Without that, the "pretrained EE50 head" advantage cannot be used in
   `flux-action` at all (see section 1).
2. On the robot, measure the TCP (wrist-roll axis to fingertip pads) and the two arm base poses, then update
   `station_geometry.yaml`. Jog each joint once to confirm signs.
3. Whichever action head is chosen, put `AnalyticIK` plus `MjCollision` (0.05 ms per frame) in the runtime safety
   filter. With the EE50 head they are required. With the joint head they still give a cheap reachability and
   collision check on predicted chunks.

## Verdict

**Kinematic gate: PASS.** EE50 end-effector targets can be executed reliably on the two 5-DoF SO-101 arms.
- On-manifold targets (FK of real robot motion, then EE50, decode, IK) come back exact: p99 position error
  1.2e-7 mm, 100% success, zero spurious joint jumps, 0.8 ms median solve per arm.
- Off-manifold error, the 6th DoF a 5-DoF arm cannot do, is removed by a well-defined projection and never
  amplified. Position is executed 1:1. Orientation noise is cut to 0.68x, because the out-of-plane component is
  dropped.
- Human handheld motion re-anchored on the robot is 98% position-feasible per 1 s chunk. Its median unexecutable
  rotation is only 0.5° (p95 3.7°).

So Simone's switch condition ("6D predictions too difficult to execute reliably") is **not** met. Kinematics does
not force the 12-D joint head.

**What this does not settle.** The gate shows EE50 is *executable*, not that it is the *better* head. The risks of
the EE50 path are on the model side, not in kinematics:
- The release has no EE50 code.
- `flux-action` normalizes with per-channel q01/q99, not per-offset z-scores.
- Reusing the pretrained EE50 head needs its exact tensor name (gated).
- The state and conditioning convention for EE50 embodiments is unknown.

BFL's own DROID comparison favored new joint heads at large batch (40.13% vs 37.88%) and EE50 only for very small
data or step budgets (bfl.ai model page). The model-side choice belongs to ws4-flux-dual.

From this workstream: with roughly 20 hours left, the joint head is the lower-risk default, and EE50 is a
contained experiment. Everything EE50 needs on the robot side now exists and is tested here. If the model side
picks EE50, the robot side is not the blocker.

## 1. EE50 convention: what is known, with evidence

Sources: the bfl.ai model page https://bfl.ai/models/flux-3-action (read 2026-10-17), the base model card
https://huggingface.co/black-forest-labs/flux-3-action-base, and the code
black-forest-labs/flux-action@830f1ce178d7aea52f18fde31d8e8e36ce6bae21.

| Item | Finding | Evidence | Status |
|---|---|---|---|
| Layout | 50 dims: first 25 left/primary, last 25 right/secondary | model page | verified (doc) |
| Per 25-D block | 3 translation, 6 rotation, 16 hand angles or gripper | model page | verified (doc) |
| 6D rotation | Zhou et al. 2019 but the first two **rows** of R. Codec: `[R00 R01 R02 R10 R11 R12]`, decoded by Gram-Schmidt on rows | model page; `ee50.rot_to_6d` / `rot_from_6d`, tested | doc verified, flattening order UNVERIFIED |
| Gripper | 0 closed, 1 open, absolute (not relative) | model page | verified (doc) |
| Gripper slot | which of the 16 aux dims | not documented, not in code | UNVERIFIED (codec default: index 9, the first aux dim) |
| Padding | unused dims zero | model page | verified (doc) |
| Anchor | the action state closest to the last conditioning *video frame*; every other pose is relative to it | model page | verified (doc). "Frame" there is a time step, not a coordinate frame |
| Coordinate frame of the relative pose | anchor end-effector frame (`T_a^-1 T_t`) or robot axes (`p_t - p_a`, `R_t R_a^T`) | not documented, not in code | UNVERIFIED. Codec supports both (`EE50Layout.rel_frame`), default `anchor` |
| Translation units | not documented | | UNVERIFIED. Codec uses metres. Per-embodiment z-scores absorb any scale |
| Normalization | z-score per embodiment, per dim, indexed by signed offset from the anchor | model page | doc verified. `EE50Stats` implements mean/std of shape (offsets, 50). Std floor is UNVERIFIED |
| Released code | **no EE50 implementation**: no rotation code, no anchor composition, no z-scores | grep over flux-action@830f1ce. Normalization is q01/q99 range scaling per channel (`src/flux_action/config.py:69-76`, `src/flux_action/data/lerobot/index.py:610-646`). Actions are `absolute` or `joint_delta` integrated onto the state (`config.py:62-68`, `policy.py:458-465`) | verified (code) |
| Reusing the EE50 head | a fine-tune keeps only the heads named after `action_modality`. Other `emb_in.action_prediction*` / `final_layer.action_prediction*` heads are dropped as "another embodiment's head" | `src/flux_action/models/wiring.py:118-143` (line 132) | verified (code). The EE50 head's name in the base checkpoint is UNVERIFIED (gated repo; the model card says 457 tensors with "ee50/gaming action heads") |
| Mixed widths | narrower actions can share one head via zero padding plus `action_mask` | `docs/embodiments.md:23` | verified (doc). For two SO-101 arms, 20 of 50 dims are active |
| 14-D joint datasets | some midtraining robot data used 7 + 7 joints (6 joints + gripper per arm) | model page | doc verified. Whether the base ships a 14-D head is UNVERIFIED (the card lists only ee50/gaming) |

**Consequence for a fine-tune that reuses the EE50 head in `flux-action`.** All of the following are needed:
- `action_modality` set to the EE50 head's exact name, and `action_dim = 50`.
- Data pre-normalized per offset by our own code, with `action_normalization: null` (identity).
- An `action_mask` over the 20 active dims.
- Custom inference post-processing: per-offset denormalize, then SE(3) composition onto the anchor, then IK.

None of this exists in the release. `ws4/kinematics/ee50.py` provides the data side and the robot side.

## 2. Kinematics (`so101_kinematics.py`)

- **Chain.** TheRobotStudio/SO-ARM100@a758567c, `Simulation/SO101/so101_new_calib.urdf`:
  - joints: lines 434-439 (pan), 414-419 (lift), 394-399 (elbow), 373-378 (wrist flex), 353-358 (wrist roll)
  - flange `gripper_frame_link`: lines 302-306
  - The constants are copied with citations, and a test re-parses the cached URDF.
  - FK matches MuJoCo on `so101_new_calib.xml` to 4e-6 m over 100 random poses. The MJCF site uses different axes
    than the URDF frame; the offset is constant.
- **Units.** "New calibration" puts each joint's zero at mid-range (`Simulation/SO101/README.md`). That is
  LeRobot's `MotorNormMode.DEGREES` zero (`lerobot==0.5.1`, `motors/motors_bus.py:858-861`). LeRobot's own
  kinematics feeds `np.deg2rad(degrees)` into this URDF (`lerobot/model/kinematics.py:66-71`), so
  `q_rad = deg2rad(q_lerobot_deg)`, tested.
  - Older recordings use `RANGE_M100_100`. For these, `lerobot_m100_to_rad` uses half the calibrated range of
    Oscar's follower (`~/.cache/huggingface/lerobot/calibration/robots/so_follower/haller_follower.json`). This is
    approximate by construction.
  - Gripper: LeRobot 0..100 maps to an EE50 open fraction of 0..1.
- **TCP.**
  - LeRobot's default target frame is the stock `gripper_frame_link`, whose z is the approach axis, 7.9 mm off
    the roll axis.
  - For the fin-ray fingers, `Station.load()` takes `tcp.pos_in_gripper = (0, 0, -0.140)` from
    `station_geometry.yaml` (ws4-sim; finger length 123.4 mm from the XLeRobot finger CAD). That point is on the
    roll axis, 41.9 mm further out than the stock flange.
  - It is marked `verified: false` and needs a measurement.
- **Station frame and mounts.** These come from `ws4/sim/station_geometry.yaml` (ws4-sim schema v1):
  - origin on the table under the torso centre, x forward, y left
  - bases at (-0.0216, ±0.1366, 0.085) m, both facing +x
  - torso box ±0.061 × ±0.193 × 0.083 m

  These values come from the XLeRobot 0.4.0 arm-base CAD and are unverified. The XLeRobot URDF
  (Vector-Wangel/XLeRobot@b017b5e `simulation/xlerobot/xlerobot.urdf`) puts the pan axes 0.31 m apart; the CAD
  gives 0.27 m.

## 3. IK method (`ik.py`)

**Structure.** By the product of exponentials, checked numerically to 1e-4:

`R(q) = Rz(-q1) · Ry(q2 + q3 + q4) · Rx(-q5) · R0`

So at a given pan angle the reachable orientations form a 2-parameter family: the approach axis must lie in the
arm's vertical plane, and the wrist roll is free.
- A 6-DoF target loses exactly one DoF: the yaw of the approach out of the arm plane.
- Straight-down grasps are reachable at any yaw, limited only by the wrist-roll range (320° of 360°).

**AnalyticIK (default).**
1. Pan from the target position. The lateral TCP offset is handled by a converged fixed point. Both pan branches
   are tried (forward, and back over the shoulder).
2. Projection of the target orientation onto the family at that pan. This is the Frobenius-closest rotation,
   found by alternating closed-form updates started from the approach-axis projection.
3. Wrist roll clamped to its range.
4. Planar 2-link solve for lift and elbow. The XLeRobot author's 2-link IK uses the same link lengths
   (0.1159/0.1350 m; XLeRobot@b017b5e `simulation/Maniskill/examples/demo_ctrl_action_ee_keyboard.py:118-175`),
   but it has no pan coupling, orientation or roll.
5. Wrist flex from the projected pitch.
6. If the projected pitch is incompatible with the position or the joint limits: a vectorized 1-D scan over the
   wrist pitch with the best roll per pitch, then bisection to the feasibility boundary. Position stays exact and
   orientation is the closest feasible.
7. Final polish: task-priority steps accepted only if they reduce position error.

Choices that matter on the robot:
- **Elbow branch of the data only** (`elbow_up=True`). In 4 of 5 datasets 100% of frames use it, and 77% in the
  fifth. Allowing both branches let noisy targets drift onto the other one and later force a flip.
- **30 Hz continuity.** Exact solutions within 0.3 rad of the previous command are preferred. For unreachable
  targets, a local solution from the previous joints is offered, so tracking degrades smoothly instead of jumping.

**DLSIK (reference).** Task-priority damped least squares: position first, orientation in the position null
space, with:
- the orientation projected onto the family at each step (without that, the unreachable component hits a
  near-zero singular value and the step destroys the position)
- an active set for joint limits
- a backtracking line search

It agrees with the analytic solver on the orientation it reaches (median difference < 0.1°). It needs a warm
start: from zero it misses (> 1 mm) on 18.5% of random reachable targets, the analytic solver on 0.5%, all of
them within 2 cm of the pan axis, with the TCP over the base.

**Timing.** Single core, real trajectory, numpy, no compilation.

| Solver | median | p99 | both arms p99 |
|---|---|---|---|
| AnalyticIK | 0.81 ms | 4.3 ms | 8.5 ms (26% of a 33 ms tick) |
| DLSIK (warm) | 0.98 ms | 1.4 ms | 2.7 ms |

The analytic tail comes from parked frames, where a joint rests on its stop and the 1-D scan runs (median 2.7 ms,
p99 6.9 ms). Active frames: median 0.75 ms, p99 1.4 ms.

## 4. Validation: thresholds and results

Thresholds were proposed before running (`validate.THRESHOLDS`). Status: ✅ pass, ❌ fail.

### (a) Real robot joint trajectories: q → FK → EE50 (float32) → decode → IK → q'

25,168 frames (each dataset capped at 6,000):
- `lerobot/svla_so101_pickplace` (SO-101)
- `Zekai-Chen/bi_so101_fold_towel_4cam` (bimanual SO-101)
- `xlerobot-team/xlerobot-left-pick-cup-sep2-20260902-1819`, `saivishwak/xlerobot-vr-teleop-medicine-bowl` and
  `Odog16/xlerobot_generalist_v2` (XLeRobot)

Each dataset contributes only `meta/info.json` plus one data parquet, in `ws4/.cache/kin/hub`. The fin-ray TCP is
used throughout. Joints outside the URDF limits are clipped first: 39% of SO-101 frames and 82% of Odog16 frames,
almost all shoulder_lift resting at -101° to -108° against a -100° URDF limit (see section 5).

| Metric | Threshold | AnalyticIK | DLSIK | |
|---|---|---|---|---|
| codec round trip (float32) | < 1e-6 m | 1.6e-8 m | | ✅ |
| position error, median | < 0.1 mm | 3.7e-8 mm | 1e-9 mm | ✅ |
| position error, p99 / max | < 1 mm | 1.2e-7 / 9.9e-5 mm | 4e-8 / 2.7e-6 mm | ✅ |
| orientation error, p99 / max | | 0.0003° / 0.011° | 0.0000° / 0.0006° | |
| success (pos < 5 mm and rot < 1°) | > 99.5% | 100% | 100% | ✅ |
| spurious jumps (> 0.3 rad per frame) | < 0.1% | 0 | 0 | ✅ |
| solve time p99, one core | < 5 ms per arm | 4.3 ms | 1.4 ms | ✅ |

4.6% of frames come back on an equivalent configuration that differs by > 0.01 rad but has the same pose. These
are parked frames where a clipped joint sits on its stop. Plot: `ws4/kinematics/plots/a_roundtrip.png`.

### (b) Human handheld and synthetic 6-DoF motion, re-anchored on robot poses

UMI handheld-gripper data (`lerobot/umi_cup_in_the_wild`, 10 Hz, resampled to 30 Hz) is cut into 1.07 s chunks
(32 actions). Each chunk is expressed in its anchor gripper frame and applied to real robot anchor poses: this is
what an EE50 policy trained on human data would ask for.
- The 7-D state layout (position, rotation vector, gripper width) is UNVERIFIED beyond sanity checks.
- Episodes with SLAM jumps (> 0.8 m/s between frames) are dropped.

The synthetic set is isotropic smooth 6-DoF motion: 5 cm and 25° median per chunk, much rotation-heavier than
human manipulation.

| Metric | UMI (human) | synthetic isotropic |
|---|---|---|
| motion per chunk, median (translation / rotation) | 2.5 cm / 2.4° | 5.1 cm / 25.5° |
| chunks position-feasible (all frames < 5 mm) | 97.9% | 97.1% |
| chunks feasible with pos < 5 mm and rot < 10° | 92.8% | 32.1% |
| unexecutable rotation, median / p95 / p99 | 0.48° / 3.7° / 7.4° | 4.4° / 24.0° / 36.8° |
| frames at a joint limit | 4.5% | 8.0% |
| jumps > 0.3 rad per frame | 0.03% | 0.04% |

Human manipulation is mostly in-plane: the 5-DoF loss is small. Infeasible chunks are workspace exits from anchors
near the edge, not orientation failures. Plot: `plots/b_irreducible_rotation.png`.

**Table workspace** (fin-ray TCP, 2 cm grid, torso footprint excluded; plot `plots/b_workspace.png`):

| Grasp | left arm | right arm | both arms |
|---|---|---|---|
| top-down, 3 cm above table, any yaw | 0.129 m² (to x ≈ 0.30 m) | 0.129 m² | 0.041 m² |
| 45° down along the radial line, 5 cm | 0.062 m² (x 0.16 to 0.42 m) | 0.062 m² | 0.010 m² |
| any orientation, 3 cm | 0.228 m² | 0.250 m² | 0.122 m² |

Bimanual work (hand-over, obstacle removal plus retrieval) must happen in the green central region, about 0.06 to
0.30 m in front of the torso and within ±0.1 m of the centre line. Top-down grasps beyond x ≈ 0.30 m are out of
reach; angled grasps reach 0.42 m.

### (c) How much off-manifold error is tolerable

FK targets from real trajectories (8,981 frames) get smooth, policy-like noise added:
- Gaussian, temporally correlated over about 5 frames
- per-axis σ in position and rotation
- isotropic, so roughly one third of the rotation noise is off-manifold

Tracked at 30 Hz with warm start. "Parked" means the source joint sits on a limit (54% of these frames, arms
resting). Failure rates exclude parked frames; full rates are in `results.json`.

| per-axis σ (pos, rot) | injected noise, median | executed vs clean, median | rot ratio | jumps (active) | pos fails (active) | check |
|---|---|---|---|---|---|---|
| 2 mm, 2° | 3.2 mm, 3.1° | 3.2 mm, 2.1° | 0.69 | 0 | 0 | ✅ |
| 5 mm, 5° | 8.0 mm, 7.8° | 8.0 mm, 5.3° | 0.68 | 0.04% | 0 | ✅ |
| 10 mm, 10° | 16.1 mm, 15.5° | 16.0 mm, 10.5° | 0.68 | 0.19% | 0 | ✅ |
| 5 mm, 20° | 8.0 mm, 31.1° | 8.0 mm, 19.7° | 0.63 | 0.50% | 0 | ❌ (jumps at the 0.5% limit) |
| 20 mm, 20° | 32.1 mm, 31.1° | 32.0 mm, 21.5° | 0.69 | 0.84% | 1.4% | ❌ |

Reading:
1. Position error passes through unchanged: the IK executes the position exactly (p99 under 0.1 µm up to 10 mm
   noise), so policy position error equals execution position error.
2. The projection removes about 32% of the orientation error, the out-of-plane part, so the executed orientation
   is **closer to the clean pose than the prediction was**.
3. Jumps stay below 0.2% of active frames up to about 15° total rotation noise. They happen near the wrist-flex
   and roll limits and at the stretched-arm edge.
4. Joint jerk (RMS of second differences) grows 1.2x at 2 mm/2° and 2.5x at 10 mm/10°. That is the noise itself,
   not the IK.

**Tolerance**: up to about 16 mm / 15° total error per step, EE50 predictions execute with no amplification and
< 0.2% branch jumps. A fine-tuned policy whose chunks are smooth and on-manifold to that level is safe to execute.
A per-step joint-velocity limiter is still required for the rare flips (it belongs in the safety layer anyway).

Plot: `plots/c_noise_tolerance.png`.

### (d) Collisions (MuJoCo mesh contacts on `ws4/sim/scenes/obstacle_target.xml`, penetration > 2 mm)

The model (read-only) is generated by ws4-sim from `station_geometry.yaml`. Contacts use the convex hulls of the
SO-101 collision meshes, the fin-ray fingers, the torso and the table, at 0.05 ms per frame.

| Dataset replayed on our station | frames | inter-arm | arm-torso | arm-table | self |
|---|---|---|---|---|---|
| Odog16/xlerobot_generalist_v2 (XLeRobot) | 20,000 | 0 | 0 | right 6.1% (≤ 54 mm) | 0 |
| Zekai-Chen/bi_so101_fold_towel_4cam | 11,634 | 0 | 0 | 0 | left 18%, right 6% (≤ 10 mm) |
| saivishwak/xlerobot-vr-teleop-medicine-bowl | 584 | 21% (≤ 29 mm) | left 5% | left 10% | 0 |

These measure geometry mismatch, not IK quality. The datasets were recorded on other mounts and tables:
- Odog16's right arm reaches below our table plane.
- saivishwak's right arm has a different mount or sign convention.
- The self contacts are convex-hull artifacts in the folded rest pose, which the real arm reaches without
  touching.

The IK adds no collision risk of its own: it stays on the data's elbow branch and executes exact positions.

Practical rule: run `MjCollision.contacts` on every IK output before sending it. Inter-arm contact is the real risk
in the shared central region, where the bimanual task happens.

## 5. What this means for the fine-tune

**Data conversion, if the EE50 head is chosen** (`ee50.windows` plus `EE50Stats.fit`):
1. Use the joint *action* stream (the leader command; EE50 anchors on the action state).
2. Convert LeRobot units to radians, choosing degrees or M100 **per dataset**.
3. Clip to the joint limits. Real arms rest a few degrees past the URDF's ±100° shoulder_lift.
4. Run FK **with the TCP of the gripper that recorded the data**: the stock flange for public SO-101 data, the
   fin-ray TCP for ours. Translations differ by the 42 mm lever arm; rotations are the same.
5. Take the poses in each arm's own base frame. With `rel_frame="anchor"`, mount errors do not matter at all.
6. Build EE50 windows at the policy's offsets.
7. Fit per-offset z-scores on our embodiment's data. At offset +32 (1.07 s) the translation std is 5 to 11 cm on
   SO-101 pick-and-place and 1 to 3 cm on the mostly idle XLeRobot generalist set.
8. Mask the 30 inactive dims.

FK errors cancel in our own round trip (the same model encodes and decodes). They only matter when mixing human or
other-robot data, where metric meaning has to agree.

**Robot side, per chunk** (example in `ws4/kinematics/README.md`):
1. Denormalize per offset.
2. Decode relative to the anchor, which is the FK of the last commanded joints.
3. `AnalyticIK` warm-started from the previous command.
4. Joint-limit and velocity clamp, then the collision check, then conversion to LeRobot degrees.

Cost: under 1 ms median per arm per step.

**Do not mix arm conventions blindly.** Replaying the XLeRobot VR dataset on our station collides in 21% of frames:
public XLeRobot recordings differ in mounting and, possibly, right-arm signs. Check a few frames visually in sim
before adding any external dataset to either head's training set.

**Joint-head alternative.** Nothing in this report argues against the native joint head on kinematic grounds; it
needs no IK in the loop. The same `Station` FK and `MjCollision` are useful as a safety filter on its predicted
chunks.

## 6. Reproduce

```bash
python -m pytest ws4/kinematics/tests -q          # 31 tests, CPU only
cd ws4 && python -I -m kinematics.hub_data        # one-off downloads into ws4/.cache/kin/hub
cd .. && python -I ws4/kinematics/validate.py     # results.json and plots/
```

Repos read (cloned into `ws4/.cache/kin/`, never executed):
- black-forest-labs/flux-action@830f1ce
- TheRobotStudio/SO-ARM100@a758567c
- Vector-Wangel/XLeRobot@b017b5e

The XLeRobot author's Notion note on analytical IK did not load (JavaScript-only page). The same author's 2-link
IK code in the XLeRobot repo was read instead (section 3).
