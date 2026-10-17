"""One minimal live call per Claude feature the orchestrator uses, with real latency, tokens and cost.

    cd ws4 && python -m orchestrator.live_check          # about USD 0.10 to 0.30; needs the API key (claude.py)

Calls, all on rendered frames of the pack-box sim at 640x480 (414 image tokens each):
1. planner, Opus 5.5, structured output + head image, effort medium (cold: writes the system-prompt cache)
2. the same request again (warm: should read the cached system prompt)
3. vision checks with Haiku 5.5 at effort low, adaptive thinking: "apple inside the box?" on a frame where it is not,
   then on a frame after the scripted stand-in dropped it in (expected: no, then yes)
4. the same two checks with Haiku 5.5 thinking disabled (latency comparison)
5. a Sonnet 5.5 lid check (lid still on the table: expected no)
Writes ``runs/live_claude_<time>.json``. Prints only usage numbers, never the key.
"""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import datetime
from typing import Any

from . import PACKAGE_DIR
from .claude import SDK_VERSION, ClaudeJSON, sdk_version
from .cost import CostMeter
from .plan import Step, concurrent_pairs
from .planner import ClaudePlanner
from .skills import load_vocabulary
from .verifier import ClaudeVerifier

GOAL = "put the apple and the yellow block in the box, then close the lid"
HEAD = "observation.images.head"


def _frames(vocab) -> tuple[dict, dict]:
    """(before, after) observations: apple on the table, then dropped in the box by the scripted stand-in."""
    from flux.dual_arm_runner import DualArmRunner, per_arm_target

    from .sim_backend import PackBoxSim, ScriptedSimBackend, SimRobot

    sim = PackBoxSim(image_size=(480, 640), randomize=False)
    robot = SimRobot(sim, seed=0)
    before = {k: v for k, v in robot.get_observation().items()}
    backend = ScriptedSimBackend(sim)
    target = dataclasses.replace(per_arm_target("left", "sim"), name="left:sim")
    runner = DualArmRunner(robot, [target], {"sim": backend})
    backend.begin("left:sim", "left", vocab["put_apple_in_box"])
    runner.start("left:sim", vocab["put_apple_in_box"].prompt)
    for _ in range(600):
        runner.step()
        if backend.finished("left:sim", robot.get_observation()["observation.state"][:6]):
            break
    after = robot.get_observation()
    truth = (sim.predicate({"in_box": "apple"}, before)[0], sim.predicate({"in_box": "apple"}, after)[0])
    runner.close()
    sim.env.close()
    return {**before, "_truth_apple_in_box": truth[0]}, {**after, "_truth_apple_in_box": truth[1]}


def run(budget_usd: float = 0.5, client: Any = None) -> dict[str, Any]:
    vocab = load_vocabulary()
    meter = CostMeter(budget_usd)
    llm = ClaudeJSON(meter, client=client)
    before, after = _frames(vocab)
    rows: list[dict[str, Any]] = []

    def last(name: str, expected: Any = None, got: Any = None, extra: dict | None = None) -> None:
        c = meter.calls[-1]
        rows.append({"call": name, "model": c.model, "latency_s": round(c.latency_s, 2), "input_tokens": c.input_tokens,
                     "cache_write_tokens": c.cache_write_tokens, "cache_read_tokens": c.cache_read_tokens,
                     "output_tokens": c.output_tokens, "thinking_tokens": c.thinking_tokens,
                     "cost_usd": round(c.cost_usd, 6), "expected": expected, "got": got, "note": c.note,
                     **(extra or {})})

    planner = ClaudePlanner(vocab, llm)
    for name in ("plan (cold cache)", "plan (warm cache)"):
        result = planner.plan(GOAL, image=before[HEAD])
        plan = result.plan
        last(name, "valid plan", "valid" if result.ok else f"invalid: {result.errors}",
             {"attempts": result.attempts, "concurrent_pairs": concurrent_pairs(plan) if plan else None,
              "steps": [(s.id, s.arm, s.skill, s.depends_on) for s in plan.steps] if plan else None})

    apple = Step("apple", "left", "put_apple_in_box", success_check=vocab["put_apple_in_box"].success_check)
    for thinking in ("adaptive", "disabled"):
        verifier = ClaudeVerifier(llm, effort="low", haiku_thinking=thinking)
        for label, obs in (("apple on table", before), ("apple in box", after)):
            v = verifier.check(apple, vocab["put_apple_in_box"], obs)
            last(f"verify haiku {thinking}: {label}", obs["_truth_apple_in_box"],
                 None if v.error else (v.done and v.confidence >= 0.7),
                 {"confidence": v.confidence, "observation": v.observation, "error": v.error})
    lid = Step("lid", "right", "close_lid", success_check=vocab["close_lid"].success_check)
    v = ClaudeVerifier(llm, effort="low").check(lid, vocab["close_lid"], after)
    last("verify sonnet: lid still on the table", False, None if v.error else (v.done and v.confidence >= 0.7),
         {"confidence": v.confidence, "observation": v.observation, "error": v.error})

    report = {"time": datetime.now().isoformat(timespec="seconds"), "sdk": sdk_version(), "sdk_pinned": SDK_VERSION,
              "image": "640x480 JPEG q90 (414 tokens)", "calls": rows, "summary": meter.summary()}
    out = PACKAGE_DIR / "runs" / f"live_claude_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    report["path"] = str(out)
    return report


def main() -> None:
    start = time.time()
    report = run()
    for r in report["calls"]:
        ok = "" if r["expected"] in (None, "valid plan") else ("OK " if r["expected"] == r["got"] else "MISMATCH ")
        print(f"{ok}{r['call']:<40} {r['model']:<18} {r['latency_s']:>6.2f} s  in {r['input_tokens']:>5} "
              f"cw {r['cache_write_tokens']:>5} cr {r['cache_read_tokens']:>5} out {r['output_tokens']:>5} "
              f"(think {r['thinking_tokens']:>4})  USD {r['cost_usd']:.5f}  got={r['got']}")
    print(f"total USD {report['summary']['total_usd']:.4f} in {time.time() - start:.0f} s -> {report['path']}")


if __name__ == "__main__":
    main()
