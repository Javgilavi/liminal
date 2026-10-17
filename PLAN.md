# PLAN.md

The strategy and the key technical decisions. [TASK.md](TASK.md) is the hour-by-hour
breakdown per workstream; [docs/architecture.md](docs/architecture.md) is the design.

## Goal

Two SO-101 arms (later the bimanual XLeRobot) that **depend on each other**. One arm
changes or maintains a condition in the scene so the other can finish a task it
couldn't do alone. Long-horizon tasks are broken into verifiable subtasks, coordinated
by a behavior tree, and executed with FLUX 3 Action as the learned manipulation backend.

**Demos:**
- **A, cooperative holding:** the left arm opens and holds a lid, the right arm inserts a block,
  and the left releases only after the insert is verified (`lid_hold_insert`).
- **B, long-horizon dependency:** the target is blocked, so one arm clears the obstacle and
  the other picks the target (`clear_obstacle_pick`, then `tidy_table_long_horizon`).

## Key decision: how FLUX drives two arms

The released FLUX 3 Action checkpoint is **single-arm** (6-D actions, scene + wrist
camera). There are two ways to get two arms:

### Option A: two embodied FLUX policies, one per arm

Each arm runs its own FLUX policy over its own 6-D action slice, wrist camera and
history. The behavior tree does all the coordination (who moves, who holds, what must
stay true).

| Variant | How | VRAM | Inference | Config |
|---------|-----|------|-----------|--------|
| **A1 shared** | one model, two per-arm sessions, single-flight lock. Optionally a different LoRA per arm | ~1× (~32 GB BF16, VERIFY) | one arm at a time; the other holds | `instances: shared`, `arm_adapters: {left: ..., right: ...}` |
| **A2 per-arm instances** | two model copies, one per arm, each with its own lock | ~2× (VERIFY) | both arms can run FLUX at once | `instances: per_arm`, `devices: {left: cuda:0, right: cuda:1}` |

- **Works with the released checkpoint today**, with no action-head changes.
- Each arm's LoRA uses ordinary single-arm demos (WS1's pipeline, 50–100 per skill).
- The two arms see the scene from different sides. A per-arm LoRA (A1 + `arm_adapters`)
  absorbs that viewpoint difference.
- Learned bimanual coordination is impossible: handovers or both arms touching one
  object at the same time have to be scripted as tree structure (WHILE / parallel).
- In A1, the single-flight lock means one arm moves under FLUX while the other is in
  position hold. That's fine for "hold + act" demos, but a bottleneck for truly simultaneous motion.

### Option B: one new dual-arm model, fine-tuned on both arms together

One FLUX model fine-tuned on a **12-D action** `[left 6 | right 6]`, conditioned on
the scene camera and both wrist cameras. Skills target the arm group `both`. The model
outputs one action vector, which is split back per arm.

- **Learned coordination:** the model can learn timing between arms (hold while
  inserting, handovers, bimanual carry) from data.
- **Needs bimanual demos:** two leader arms teleoperating together. Expect more episodes
  than single-arm LoRA (hundreds rather than tens; set the target after the first training curve).
- **Needs an adapted action head / normalisation for 12-D.** Check with the BFL docs
  whether the checkpoint supports a different action dimension or needs a new head
  (VERIFY). This is the biggest unknown.
- One inference call drives both arms, so there's no single-flight contention between arms.
- Skills become fewer and larger (e.g. `open_lid_and_insert`). The tree still
  sequences, verifies and recovers around them.
- The same recipe extends to XLeRobot (`configs/embodiment/xlerobot.yaml`, head and base
  excluded from the policy at first).

### Comparison

| | A1 shared | A2 per-arm instances | B bimanual fine-tune |
|---|---|---|---|
| Works with released checkpoint | **yes** | **yes** | no, needs new fine-tune |
| Data needed | single-arm demos | single-arm demos | bimanual demos (more) |
| Learned inter-arm coordination | no | no | **yes** |
| Simultaneous FLUX motion on both arms | no | **yes** | **yes** |
| VRAM | ~1× | ~2× | ~1× |
| Hackathon risk | **low** | medium (hardware) | high (head + data + training) |
| Who | WS1 | WS1 + WS5 (GPUs) | WS4 (+ WS1 recipe) |

### Decision

1. **Ship the demos on Option A1** (`configs/policy/flux3_so101.yaml`). It's what the
   architecture assumes, needs no new model, and the hold-while-acting demo doesn't need
   simultaneous FLUX motion. Upgrade to **A2** only if we get a second GPU and a demo
   needs both arms moving under FLUX at once.
2. **Run Option B as the research track** in WS4 (`configs/policy/flux3_dual_bimanual.yaml`).
   Start with sim + open bimanual datasets, then our own bimanual demos.
3. **Go/no-go for B at hour 18:** a trained 12-D checkpoint that beats A1 on
   `lid_hold_insert` in sim → try it on hardware with `lid_hold_insert_bimanual`.
   Otherwise B is presented as future work with its training curve.

Both options run on the same executor, safety layer and task grammar. Switching is a
config change plus the matching task file:

```bash
# Option A (per-arm): skills on `left` / `right`
liminal run configs/tasks/lid_hold_insert.yaml --robot configs/robot/so101_dual.yaml \
    --backend flux --policy configs/policy/flux3_so101.yaml
# Option B (bimanual): one skill on `both`
liminal run configs/tasks/lid_hold_insert_bimanual.yaml --robot configs/robot/so101_dual.yaml \
    --backend flux --policy configs/policy/flux3_dual_bimanual.yaml
```

### A vs B experiment (same protocol for both)

- Tasks: `lid_hold_insert` (A) vs `lid_hold_insert_bimanual` (B), same layouts, same props.
- 10 trials × 3 layouts each, in sim first, then on hardware.
- Metrics: success rate, time to goal, number of recoveries (from `runs/*.jsonl`), lid-slip events.
- Data cost: demos and training hours needed to reach that success rate.

## Phases

| Phase | Hours | Outcome |
|-------|-------|---------|
| 0. Contracts | 0–2 | interfaces, predicates, task grammar agreed; `pytest` green everywhere |
| 1. Bring-up | 2–6 | arms + cameras calibrated; FLUX inference measured; sim scene loads |
| 2. First skills | 6–12 | one FLUX skill on hardware; dual-arm task end-to-end in mock + sim |
| 3. Collaboration | 12–18 | physical WHILE (hold + insert); demo LoRA trained; **B go/no-go** |
| 4. Long horizon | 18–24 | obstacle-clearing + long-horizon task with recovery on hardware |
| 5. Freeze + demo | 24–30 | ≥5 trials per demo, success rates, video, slides |

## Top risks

| Risk | Mitigation |
|------|-----------|
| FLUX not running by hour 6 | scripted waypoint backend; team never blocks on GPU |
| Not enough VRAM | A1 (single model) instead of A2; smaller batch/precision per BFL docs |
| Out-of-the-box FLUX poor on our table | WS1 LoRA on 50–100 demos with a fixed eval protocol |
| Lid slips (position-only hold) | light lid, short holds, WHILE monitor cancels + retries |
| Option B action head can't be changed easily | B stays a research track; demos don't depend on it |
| Perception false positives | cheap colour zones first, VLM verifier only for hard predicates, human check on trials |
