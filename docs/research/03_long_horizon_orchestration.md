# WS3: Long-horizon decomposition and orchestration

## Decision so far

**A behavior tree enforces execution; LLMs/VLMs only propose trees.** Every planner
emits the task-file grammar, which is compiled and validated (`task_spec.py`). Compile
errors go back to the model for self-repair (`LLMPlanner`).

## Options compared

| Approach | Strengths | Weaknesses | Role here |
|----------|-----------|-----------|-----------|
| Hand-written BT (`TemplatePlanner`) | reliable, debuggable, demo-safe | no generalisation | baseline + demo path |
| Finite state machine | simple, explicit transitions | concurrency (WHILE) and recovery explode the state count | comparison baseline only |
| Frontier LLM → BT (Claude etc.) | generalises to new goals, good at decomposition | can hallucinate skills or predicates (the compiler catches those), latency | `LLMPlanner(ClaudeModel)` |
| Gemini Robotics-ER (latest) → BT | spatial grounding from images (points, boxes), embodied reasoning | provider-specific; spatial output needs mapping to our frame | `LLMPlanner(GeminiModel)` + perception helper |
| VLM as verifier | flexible predicates ("is the block in the box?") | slow, can be wrong | `VLMVerifier`, on demand only |
| Single end-to-end policy | no decomposition needed | needs long-horizon data we don't have | out of scope |

## Design rules

- Every subtask has a **verifiable** success predicate. No predicate, no subtask.
- Use `retry` around fragile manipulation and `fallback` for alternatives.
- Subgoal markers (`set:`) + `fallback: [check: x, ...]` make tasks resumable.
- Replanning input = goal + current world facts + recent event log, never raw video.

## Bake-off (H6–12)

Goals: (1) put block in bowl, (2) put block in the box with the lid, (3) block behind
cup into bowl, (4) two blocks into the box, (5) "tidy the table".
For each planner: valid-tree rate over 5 samples, manual correctness, latency, and
whether WHILE was used where needed. Record results here.

| Planner | valid % | correct % | latency | notes |
|---------|---------|-----------|---------|-------|
| template | | | | |
| Claude | | | | |
| Gemini ER | | | | |

## References (verify before citing)

- SayCan (Ahn et al., 2022): LLM proposals grounded by skill affordances.
- Code as Policies (Liang et al., 2022); Inner Monologue (Huang et al., 2022): closed-loop feedback to the LLM.
- Behavior trees in robotics: Colledanchise & Ögren, *Behavior Trees in Robotics and AI*; py_trees; BehaviorTree.CPP.
- Gemini Robotics / Gemini Robotics-ER (Google DeepMind): embodied reasoning models.

## Live action-level hybrid exception

The standalone live runners now have an optional Claude gate at every FLUX chunk
boundary: student prefix, Cartesian edit, fixed EEF target, or stop. This is separate
from task-tree planning and does not use a recovery skill allowlist. The abandoned
skill-recovery prototype and its configuration have been removed. See
[the live hybrid contract and enablement instructions](../official-inference.md#optional-action-level-claude-hybrid-controller).
