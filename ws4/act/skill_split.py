"""Train / held-out episodes of a skill dataset, exactly as ``single_arm/build_skill.py`` splits it, as shell lines.

``ws4_split.json`` at the dataset root when it exists (ws4-vr2's split: the last 3 good takes of each box position
held out, review rejects dropped); otherwise build_skill's provisional rule (review.json rejects dropped, the last 3
good takes held out), flagged ``PROVISIONAL=1``. The red-pen split stores ``train`` as a count, so the train takes
are then every episode that is neither dropped nor held out.

    eval "$(python ws4/act/skill_split.py /workspace/act/raw/haller_open_the_box)"   # TRAIN_EPS EVAL_EPS EXCLUDE ...
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

WS4 = Path(__file__).resolve().parents[1]
if str(WS4 / "single_arm") not in sys.path:
    sys.path.insert(0, str(WS4 / "single_arm"))


def split(root: Path) -> dict:
    total = json.loads((root / "meta" / "info.json").read_text())["total_episodes"]
    episodes = list(range(total))
    path = root / "ws4_split.json"
    raw = json.loads(path.read_text()) if path.exists() else None
    if raw is not None and not isinstance(raw.get("train"), list):
        heldout = raw["heldout"] if isinstance(raw["heldout"], dict) else {"H": list(raw["heldout"])}
        held = {e for takes in heldout.values() for e in takes}
        drop = set(raw.get("drop", []))
        train = [e for e in episodes if e not in drop | held]
        out = {"source": f"ws4_split.json {raw.get('updated', '')}".strip(), "heldout": heldout,
               "drop": sorted(drop), "train": train, "provisional": False}  # fmt: skip
    else:
        from build_skill import split_of

        out = split_of(root, episodes)
    held = sorted(e for takes in out["heldout"].values() for e in takes)
    if set(held) & set(out["train"]):
        raise SystemExit(f"{root}: held-out takes {sorted(set(held) & set(out['train']))} are also train takes")
    out["eval"] = held
    out["exclude"] = [e for e in episodes if e not in held]  # offline_diag samples from the episodes NOT excluded
    return out


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: skill_split.py DATASET_ROOT")
    s = split(Path(sys.argv[1]))
    lines = {"TRAIN_EPS": " ".join(map(str, s["train"])), "EVAL_EPS": " ".join(map(str, s["eval"])),
             "EXCLUDE": " ".join(map(str, s["exclude"])), "DROP_EPS": " ".join(map(str, s["drop"])),
             "SPLIT_SOURCE": s["source"], "PROVISIONAL": int(bool(s["provisional"])),
             "HELDOUT_JSON": json.dumps(s["heldout"])}  # fmt: skip
    print("\n".join(f"{k}={shlex.quote(str(v))}" for k, v in lines.items()))


if __name__ == "__main__":
    main()
