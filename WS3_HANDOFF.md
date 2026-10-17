# WS3 harness: continuation guide for any agent

Read this first if you're picking up WS3 work (a Claude Code session, Codex, or another model).
It records what the harness is, what has been built, how to run and check it, the decisions behind
it, and what is still open. Last updated 2026-10-17 after the authorized commit batch.

## 0. Situation in one paragraph

Liminal is the hackathon's dual-arm SO-101 orchestration layer on top of FLUX 3 Action, the
mandatory manipulation VLA because FLUX owns the track. The user (Javier Gil) owns **WS3**:
long-horizon decomposition, memory and orchestration ("the harness"). The harness takes a
text prompt or a demonstration video and breaks it into atomic, verifiable FLUX skill calls per
arm. It runs them one validated step at a time, verifies every step from fresh observations,
and remembers progress across interruptions and runs. **V0–V6 and a web UI are built and
tested offline** on a replay stand-in world. Nothing has run on hardware, in MuJoCo through the
harness, or against the real Claude API in the verified harness test suite. The user explicitly
authorized committing this batch on branch `dev/harness`; changes are grouped into focused
commits and have not been pushed.

Separately, the standalone MuJoCo browser app was connected to real FLUX 3 Action on the
user's existing Runpod A100. A 64-tick right-arm test exercised two action chunks, and a
30-tick browser-API episode completed without errors. These tests verify inference and
motion, not pick-and-place success or Liminal harness integration. The local UI, SSH tunnel
and cloud FLUX process were then stopped at the user's request; the Pod remains running.
See `RUNPOD_SETUP.md` and `examples/dual_so101_sim/flux/RUNPOD_FLUX.md` for manual restart.
The original working directory `../dual_so101_sim/` is preserved, and its source snapshot
is versioned in `examples/dual_so101_sim/` without environments, weights or connection pins.

## 1. What the user asked for, and the agreed plan

- **Research** (done, gitignored files): `ME.md` covers the research and `ME_PLAN.md` the plan.
  `ME_PLAN.md` §9.2 defines the versions:
  - **V0**: verified FLUX skill
  - **V1**: authored short procedure
  - **V2**: Claude planning
  - **V3**: memory (frames, ledger, resume)
  - **V4**: video demonstration input
  - **V5**: cooperative arms
  - **V6**: measured refinements (experience retrieval, keyframe selection, evaluation)
- **Chosen design:** "Claude + Harness VLA ideas + MemER ideas + FLUX", plus the Video-to-BT
  structure for video input:
  - Harness VLA: https://arxiv.org/html/2607.08448v5
  - MemER: https://jen-pan.github.io/memer/
  - Video-to-BT: https://video2bt.github.io/video2bt_page/ (paper https://arxiv.org/html/2509.16611)
  - We borrow ideas and interfaces, not code.
- **Later request:** build everything through the "final" harness, add a frontend with option
  selection and prompt input that shows the per-arm atomic skill breakdown, make memory work
  like MemER and Harness VLA, and add Video-to-BT-style video input. Do it by orchestrating
  subagents, and test everything.
- **Earlier parallel work:** test with a small VLA (SmolVLA first, then π0.5) in the
  user's MuJoCo sim (`../dual_so101_sim`). See §7; that harness integration remains unverified.

## 2. How the harness works (the mental model)

```
text prompt / demo video / task file
        │  grounding: closed sets only (registered skills × known objects × available arms)
        ▼
ProcedureProposal  ──►  operator reviews per-arm atomic steps (UI or --plan-only), edits, accepts
        │                (server/CLI re-validates; the accepted procedure is frozen next to the ledger)
        ├── planner mode (V2/V3): loop { observe → reconcile ledger → planner proposes ONE step
        │        (current + historical frames, verified-only progress, advisory experience,
        │        operator_arms) → validate → recheck scene revision → save attempt → FLUX skill
        │        on one arm → fresh observation → verify → remember }
        └── authored mode (V1/V5): compiled reactive tree: retry{fallback[check effect, skill]};
                 a step with `maintain` becomes establish → while(hold + re-observe) → release
```

Hard rules (also in `AGENTS.md`):
- **Verification:** a step counts only when a registered observer's fresh evidence shows its
  effect. A backend or VLA "success" is never trusted. Unknown, e.g. occluded, is never done.
- **LLMs propose, trusted code decides:** the model can't add skills, objects, physical facts
  or code. The FLUX instruction text is rendered from the skill's template, not by the model.
- **Advisory memory:** memory and experience never complete obligations.
- **Safety:** every motor command goes through `SafetySupervisor`. One arm, one owner.

## 3. Code map (all paths under `liminal/`)

| Area | Files | Notes |
|---|---|---|
| Procedure contract | `src/liminal/orchestration/procedure.py` | `ProcedureProposal`, `parse_procedure` (closed-set validation), `describe` (UI rows with `flux_prompt`), `compile_procedure` (reactive and cooperative trees), `ground_goal`, `KeywordGroundingModel` (offline grounder), `scenario_with_procedure` |
| Service (CLI + UI) | `src/liminal/orchestration/service.py` | `RunOptions`, `prepare`, `execute`, `ground`, `ground_demo`, `list_options`, `MEMORY_OPTIONS`, `memory_kwargs`. Today it always builds the replay world. |
| Planner loop | `orchestration/episode.py` (`run_episode`, `run_authored`), `planner.py` (`ConstrainedPlanner`, `ClaudeModel`), `plan_validation.py` (`validate_proposal`, `operator_arms` check) | |
| Execution | `orchestration/bt.py` (`SkillNode`, `While` with evidence monitoring, `Retry`...), `attempt.py` (`AttemptScope`, `EvidenceAttempts`, `void_since`), `executor.py`, `context.py`, `task_spec.py`, `skills.py` (primitives: home/release/hold) | |
| Evidence | `perception/evidence.py`, `world_state.py` (strict three-valued facts, `active_attempt_id`), `verifier.py` | |
| Replay world | `orchestration/replay.py` (`ReplayScene`, `ReplayBackend` with `fail_first`/`lie_first`, `ReplayObserver`, `ReplayModel` (scripted planner), PNG renderer, state predicates and faults for the lid scenes), `configs/replay/*.yaml` | `workbench.yaml` is the free-text scene: 3 blocks, box, 2 bins |
| Memory | `memory/store.py` (SQLite ledger), `keyframes.py` (MemER-style event/nominated/none + clustering), `progress.py` (verified-only summary), `experience.py` (cross-run procedures and failures, compatibility filter, frozen corpus), `inspection.py` (`inspect_run`) | |
| Video input (V4) | `src/liminal/demo/` (`load_demo`, `ground_demo`, two-stage describe→map, `apply_corrections`, `bind_role`, `ReplayDemoModel` scripted, `synth.py` frame generator), `configs/demos/` | demos are synthetic |
| Evaluation (V6) | `src/liminal/eval/`, `configs/eval/{smoke,orchestration,memory_ablation}.yaml` | matched budgets, Wilson CIs, false-completion check |
| UI | `src/liminal/ui/server.py` (FastAPI, `/api/*`, kept `/run` `/estop` `/state`), `ui/web/{index.html,app.js,style.css}` (no build, no CDN), `ui/cli.py` | |
| Config | `.env.example` (`ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL=claude-haiku-5-5`, call caps), `src/liminal/settings.py` | |

Older per-worker notes: `outputs/ws3_development/{planner,evidence,memory}_handoff.md`.
Research: `outputs/ws3_research/`. Both are gitignored.

## 4. Run and check

```bash
cd liminal && source .venv/bin/activate          # venv has dev, ui, anthropic installed
uv --cache-dir /tmp/liminal-uv-cache pip install --python .venv/bin/python -e '.[dev,ui,llm]'  # if rebuilding
pytest -q                                       # 356 passed at last run
ruff check src tests scripts                    # clean

liminal serve --port 8765                       # web console at http://127.0.0.1:8765
liminal episode configs/replay/workbench.yaml --goal "put the green block in the left bin, then the red block in the right bin with the right arm" --plan-only
liminal episode configs/replay/workbench.yaml --goal "put the blue and green blocks in the box" --db runs/g.sqlite3 --interrupt-after 1
liminal episode configs/replay/workbench.yaml --db runs/g.sqlite3 --resume <run-id>
liminal episode configs/replay/workbench.yaml --goal "put red in the right bin" --authored --lie-first
liminal episode configs/replay/workbench.yaml --demo configs/demos/stack_and_hidden_demo --plan-only
liminal episode configs/replay/two_blocks.yaml --db runs/m.sqlite3 --keyframes nominated --experience readwrite
liminal episode configs/replay/lid_hold_insert_slip.yaml --task configs/tasks/replay_lid_hold_insert.yaml
liminal eval configs/eval/orchestration.yaml    # writes runs/eval/<name>/{results.json,report.md}
liminal episode ... --model claude              # needs ANTHROPIC_API_KEY in .env; never run yet
```

`runs/` and `outputs/` are gitignored. `.claude/launch.json` starts the console on port 8766 for
the agent's browser preview. Keep the user's port, 8765, free.

Last evaluation (replay, deterministic, so exploratory only):
- **Planner:** 21/21 verified, with 0 false completions.
- **Authored:** 15/18 verified. Its only miss is `fail_then_lie`: a failed grasp followed by a
  false success uses up the skill's 2-attempt cap, and the tree reports failure honestly.
- **Memory ablations:** all 6/6. The scripted model can't show memory benefits.

## 5. Decisions and why

- **One procedure format for every input**, so text, video and experience share one validator,
  one executor and one UI view.
- **Planner mode commits one step at a time.** The plan is redone after every observation,
  which gives reactivity (stale plans are dropped, undone effects reopen) without a ticked
  BT library.
- **Operator authority:** an accepted procedure's arm per effect is enforced in planner mode
  (`operator_arms`), and the accepted procedure is frozen at `{db}.{run_id}.procedure.json`
  so a resume can't re-ground differently.
- **Replay first:** replay scenes, scripted models and synthetic demos let all V0–V6 contracts be
  tested on a laptop. They are always labeled `REPLAY/MOCK, not hardware`.
- **Cheap model by default:** `claude-haiku-5-5`, capped calls, no automatic escalation (user
  preference).
- **MemER finding applied:** predicted subtasks are never stored as memory, because they hurt
  in MemER. Memory text comes only from verified ledger facts.

## 6. Known limits and open work

- **No real Claude:** the Claude path (planning, grounding, demo vision, keyframe nomination)
  has never been run because there's no key. Occlusion recall needs it.
- **No real video:** the demo videos are synthetic, and no real human video has been processed.
- **Fixed skill set:** the skills are `pick_place`, plus `open_lid`/`insert` in the lid
  scenes. "Any prompt" means compositions of those, and the rest is flagged.
- **WS1:** the FLUX loader and processors (`policy/flux.py` stubs) and per-arm sessions.
- **WS5:** the Haller gateway and the joint/camera unit contract.
- **No real camera observer.**
- **Team heads-up needed** for shared files:
  - the `AGENT.md` → `AGENTS.md` rename (`CLAUDE.md` is a copy)
  - new `ExecutionContext` fields (`attempts`, `observe`)
  - `WorldState` accessors
  - the `ui/server.py` rewrite (WS5 owns it)
  - the `operator_arms` check in `plan_validation.py`
- **Stale entries:** `configs/skills.yaml` still lists `left_holding`/`right_holding`.
- **FastAPI deprecation:** `on_event` in `ui/server.py` causes deprecation warnings.
- **Commit status:** the user authorized this batch; focused `ws<N>:` commits are on
  `dev/harness`. No push was requested or performed. Private ME notes remain gitignored.

## 7. Earlier parallel integration effort (remaining work is unverified)

- **Claude Code session "Run the harness in MuJoCo with a LeRobot VLA"** (id
  `local_879cf817-017b-4ba1-b06d-3966b3dc3280`) is building:
  - a policy-agnostic LeRobot policy server: SmolVLA first, then π0.5, with FLUX via
    `../dual_so101_sim/flux/server.py`
  - an `HttpPolicyBackend` skill backend through `SafetySupervisor`
  - a MuJoCo ground-truth observer
  - a sim scenario format
  - a `RunOptions.world="mujoco"` switch
  - a real smoke run

  Checkpoint names and action spaces are unverified. The laptop GPU is an 8 GB RTX 5070, and
  FLUX needs `--offload` or a cloud GPU. Check its result (files and test run) before relying
  on it.

  The generic `scripts/policy_server.py` and pure `src/liminal/sim/spec.py` are included in
  this commit batch. Their presence does not establish that the other proposed components
  are implemented or that a complete harness-to-MuJoCo run passed. The separate cloud FLUX
  app described in §0 is the path that was actually exercised with real FLUX.

## 8. Agent and session history

- **Codex orchestrator:** cwd `/home/jgilaviles/erh_hack`, logs in
  `~/.codex/sessions/2026/10/17/rollout-2026-10-17T13-06-19-...jsonl`, with GPT-6 Luna workers
  `v3_memory`, `v3_evidence`, `v3_planner` and research workers. It built the planner, evidence
  and memory layers, then hit its ChatGPT usage limit around 13:14.
- **Claude Code main session** (orchestrator, Opus): it took over, fixed tests and lint, built
  V1 attempts, the procedure contract, the service, the CLI wiring, the cooperative compile and
  operator arms, and did all integration and verification.
- **Sonnet 5.5 sessions** (Claude Code desktop, same checkout, disjoint files):

  | Session | Built |
  |---|---|
  | `replay-fixtures` | the replay layer |
  | `episode-runner` | `run_episode` and `liminal episode` |
  | `episode-cli-v1-task-path` | `--task`, moving-destination demo |
  | `video-procedure` | `demo/` (V4) |
  | `memory-upgrade` | keyframes, progress, experience |
  | `cooperative-arms` | V5 `While`, lid scenes, renderer |
  | `harness-web-ui` | `ui/` |
  | `eval-runner` | `eval/` |

  Their transcripts can be read from the Claude desktop app. Those activity reports belong
  to the earlier handoff; current session activity was not re-verified during committing.
- **User preferences:**
  - wants sub-agents as separately openable chats
  - prefers Sonnet 5.5 for coding sub-agents
  - wants a low-cost Claude model in the harness
  - wants explanations with screenshots and diagrams
  - previously requested review before commits; explicitly authorized this batch on 2026-10-17

## 9. How to continue (checklist for the next agent)

1. Read `AGENTS.md`, then this file, then `ME_PLAN.md` §9.2.
2. `git status` (the authorized batch is committed; investigate any new changes).
   Run `pytest -q` and `ruff check src tests scripts examples/dual_so101_sim`.
3. Check the MuJoCo/VLA session's output (§7) and verify it yourself.
4. Next priorities, unless the user says otherwise:
   - (a) a live Claude test once a key is in `.env`
   - (b) sim-in-the-loop with SmolVLA, then FLUX
   - (c) the real camera observer and FLUX/Haller integration with WS1/WS5
   - (d) review new changes separately from this already-authorized commit batch
5. Keep the rules in §2, and label every non-hardware result as such.
