"""Human review of a demo-derived proposal: remove / change arm / change destination / reorder.

The Video-to-BT paper reports refined results with human feedback. We keep first-pass and corrected
proposals separable: ``apply_corrections`` never mutates its input, re-validates the result with
``parse_procedure``, tags each touched step with an ``operator`` source and counts the corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from liminal.orchestration.procedure import ProcedureError, ProcedureProposal, SkillWorld, parse_procedure

OPS = ("remove", "arm", "destination", "reorder")


@dataclass(frozen=True)
class CorrectedProcedure:
    proposal: ProcedureProposal  # validated, after the corrections
    first_pass: ProcedureProposal  # exactly what was proposed before the operator touched it
    corrections: int  # number of operator corrections applied (0 = untouched)
    log: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"corrections": self.corrections, "log": list(self.log), "first_pass": self.first_pass.to_dict(),
                "proposal": self.proposal.to_dict()}


def apply_corrections(proposal: ProcedureProposal, corrections: list[dict[str, Any]],
                      world: SkillWorld) -> CorrectedProcedure:
    data = proposal.to_dict()
    steps: list[dict[str, Any]] = data["steps"]
    obligations = {s["effect"] for s in steps if s["effect"] in proposal.obligations}
    log: list[str] = []

    def find(step_id: Any) -> dict[str, Any]:
        for s in steps:
            if s["id"] == step_id:
                return s
        raise ProcedureError(f"correction refers to unknown step {step_id!r}")

    def tag(step: dict[str, Any], what: str) -> None:
        step["sources"].append({"kind": "operator", "ref": "correction", "note": what})

    if not isinstance(corrections, list):
        raise ProcedureError("corrections must be a list")
    for i, corr in enumerate(corrections):
        if not isinstance(corr, dict) or corr.get("op") not in OPS:
            raise ProcedureError(f"corrections[{i}]: op must be one of {OPS}")
        op = corr["op"]
        if op == "remove":
            step = find(corr.get("step"))
            steps.remove(step)
            obligations.discard(step["effect"])
            for s in steps:
                s["depends_on"] = [d for d in s["depends_on"] if d != step["id"]]
            log.append(f"removed {step['id']}")
        elif op == "arm":
            step = find(corr.get("step"))
            step["arm"] = corr.get("arm")
            tag(step, f"arm -> {corr.get('arm')}")
            log.append(f"{step['id']}: arm -> {corr.get('arm')}")
        elif op == "destination":
            step = find(corr.get("step"))
            dest = corr.get("destination_id")
            skill = world.registry.skills.get(step["skill"])
            if not isinstance(dest, str) or dest not in world.objects or skill is None:
                raise ProcedureError(f"corrections[{i}]: unknown destination {dest!r}")
            new_effect = world.effect_for(skill, step["object_id"], dest)
            if step["effect"] in obligations:
                obligations.discard(step["effect"])
                obligations.add(new_effect)
            step["destination_id"], step["effect"] = dest, new_effect
            tag(step, f"destination -> {dest}")
            log.append(f"{step['id']}: destination -> {dest}")
        else:  # reorder
            order = corr.get("order")
            if not isinstance(order, list) or sorted(map(str, order)) != sorted(s["id"] for s in steps):
                raise ProcedureError(f"corrections[{i}]: order must list every current step id exactly once")
            steps[:] = [find(sid) for sid in order]
            log.append("reordered: " + " -> ".join(order))

    data["steps"] = steps
    data["obligations"] = [s["effect"] for s in steps if s["effect"] in obligations]
    for s in steps:  # the registered effect is recomputed by the validator
        s.pop("effect")
    fixed = parse_procedure(data, world)  # raises ProcedureError if a reorder broke depends_on, etc.
    return CorrectedProcedure(fixed, proposal, len(corrections), tuple(log))
