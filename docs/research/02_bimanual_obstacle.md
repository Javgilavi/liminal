# WS2: Bimanual obstacle removal to reach a target

## Problem

The target is blocked (occluded, in the corridor, under a lid). The robot has to find
the blockers, decide **who clears what and how**, keep it cleared if needed, and then reach
the target, with verifiable predicates at every step.

## Strategy taxonomy (implemented in `bimanual/obstacle.py`)

| Strategy | When | Tree shape | Risk |
|----------|------|-----------|------|
| `push_aside` | light, stable object in the corridor | `retry(push)` → act | low precision needed, may topple tall objects |
| `pick_aside` | graspable, must leave the area | `retry(pick to free zone)` → act | grasp failures |
| `hold_away` | springs back / falls back (lid, flap, curtain) | `lift` → `while(hold) { act }` | needs two arms; position-only hold |

Role assignment baseline: minimise (clear-arm → obstacle) + (act-arm → target) distance,
prefer two different arms, and require two arms for `hold_away`. Reachability is a disc
for now. Replace it with the real SO-101 workspace (IK or sampled reachable set) later.

## Open research questions

1. **Detection of blockers:** geometric (top camera, occupancy along the reach
   corridor) vs VLM ("what blocks the red block?") vs Gemini ER-style pointing.
2. **Strategy selection:** rules on object class vs VLM choice vs try-cheapest-first with
   `fallback: [push_aside, pick_aside]`.
3. **Collision safety while holding:** the held pose plus the other arm's path. Start with
   zone locks + conservative held poses; later, swept-volume checks in sim.
4. **Re-checking:** clearing one object can expose or create another blocker. Re-run
   detection after each clear (loop until `!blocked`).
5. **Learned vs composed:** a single bimanual policy for "clear and pick" (needs joint
   data, see WS4) vs composition of single-arm FLUX skills (what we do now).

## Experiments

- E1: hand-entered scene → generated tree → mock + sim. Valid tree rate, plan correctness.
- E2: same on hardware with scripted skills. Success, collisions, time.
- E3: same with FLUX skills (`push_aside`, `pick_place`) from WS1.
- E4: perception-provided scene (WS3 detectors) instead of hand-entered.

## References (verify before citing)

- ALOHA / ACT (Zhao et al., 2023) and Mobile ALOHA (Fu et al., 2024): low-cost bimanual teleop and imitation.
- Bimanual role assignment / "stabilise + act" decompositions in the bimanual manipulation literature.
- Gemini Robotics-ER: embodied reasoning (pointing, spatial understanding) for blocker detection.
