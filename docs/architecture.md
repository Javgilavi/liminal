# Architecture

## Core idea

The hard part is not moving objects. It's one arm **changing or maintaining a
condition** so the other arm can do something otherwise impossible. Three
collaboration patterns:

| Pattern | Example | Hackathon scope |
|---------|---------|-----------------|
| Simultaneous: A maintains a condition while B works | hold lid open → insert block | **yes** (Demo A, `lid_hold_insert`) |
| Sequential: A prepares the scene, then B acts | push cup aside → pick block | **yes** (Demo B, `clear_obstacle_pick`) |
| Coordinated: both arms manipulate one object | carry a tray together | design for it, not built |

## Layers

```mermaid
flowchart TB
    UI[ui: CLI / HTTP / voice] --> P
    P[planner: template | LLM | Gemini ER | bimanual obstacle] -->|task YAML| C[task_spec compiler]
    C --> BT[behavior tree executor]
    BT <--> W[(world state + event log)]
    PER[perception: colour zones, VLM verifier, sim ground truth] --> W
    BT --> S[safety supervisor]
    S --> R[skill router]
    R -->|hold, release, home| PRIM[primitive backend]
    R -->|learned skills| POL[FLUX 3 Action / scripted / mock]
    PRIM --> A1[left arm]
    PRIM --> A2[right arm]
    POL --> A1
    POL --> A2
```

| Module | Owns | Key file |
|--------|------|----------|
| Planner | *what* has to happen | `orchestration/planner.py`, `bimanual/obstacle.py` |
| Behavior tree | *when*, in what order, under what conditions | `orchestration/bt.py` |
| World state | *what is true now*, what already happened | `perception/world_state.py` |
| Safety | *is this physically allowed* | `safety/supervisor.py` |
| Skill backends | *how* an arm moves | `policy/`, `orchestration/skills.py` |
| Arms / sim | joint I/O and position hold | `hardware/`, `sim/` |

## Dependencies → nodes

| Dependency | Meaning | Node |
|-----------|---------|------|
| AFTER | A finishes before B starts | `sequence: [A, B]` |
| WHILE | A must remain true during B | `while: {arm, maintain, do: B}` |
| UNTIL | A continues until a condition | `while: {..., until: cond}` / `wait: cond` |

`While` semantics (the bit a plain action list cannot express):

1. claim the support arm and switch it to the **local position-hold controller**
   (no VLA inference needed to keep holding)
2. wait for `maintain` to become true (bounded by `establish_timeout_s`)
3. run the body and monitor `maintain` concurrently. If it breaks, **cancel the body
   and fail** so an enclosing `retry` / `fallback` recovers
4. on body success require `until`. The hold stays on until an explicit `release`

## FLUX 3 Action integration

FLUX is a learned manipulation backend for bounded skills. It's not the coordinator.

- One model instance, two per-arm sessions. Each arm has its **own** observation
  history (scene + its wrist camera + its joint state).
- **Single-flight inference** behind one lock. While FLUX serves one arm, the other
  arm sits on its hold controller, which gives simultaneous physical collaboration
  without two concurrent model calls.
- Action chunks run through `SafetySupervisor.send()` at `control_hz`; the skill
  exits early as soon as its `success` predicate holds.
- Released checkpoint is single-arm 6-D. Bimanual options: per-arm sessions (now) vs
  a joint 12-D fine-tune (WS4 research).
- Planning-note numbers, all `VERIFY`: 7B world-action model, scene + wrist cameras,
  42-step chunks, 32 executed at 30 Hz, 8-frame history, ~32 GB VRAM in BF16, LoRA recipe ~10k steps.

## Safety model

- **Ownership:** one node commands an arm at a time. A conflict is a bug and aborts the run with an e-stop.
- **Zones:** skills declare shared zones (`zones: [box]`). Only one *moving* arm per zone.
  A holding arm doesn't take the lock, so plan holds to stay clear of the other arm's path.
- **Limits:** clip to joint limits and max per-step delta; reject NaN/inf.
- **E-stop:** CLI Ctrl-C, `POST /estop`, any `ArmConflict`/`EmergencyStop`. Freezes
  all arms in place.
- **Force:** SO-101 + FLUX give joint *position* targets, not torque. Holding is
  position holding, so use light lids and low-force contacts. Torque-regulated tasks
  (e.g. unscrewing caps) are out of scope.

## Memory

`WorldState` holds symbolic facts plus an append-only JSONL event log (facts with
source, node start/end, retries, WHILE violations). Subgoal markers (`set:`) and
`fallback: [check: done, ...]` let a long task skip work that's already done after a
restart or replan. An LLM can read the log for replanning, but correctness is enforced
by the tree and predicates.

## Simulation's role

Use simulation for **orchestration** (trees, synchronisation, zones, failure
injection, reset logic) via the same `Arm` API. Don't assume a policy trained on real
images transfers zero-shot to rendered images. `MockBackend` + `Fault` already covers a
lot of failure testing with no physics at all.
