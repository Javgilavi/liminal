"""Write a skill's split as shell variables (skill.env) from a dataset's final ws4_split.json (ws4/cloud/skill_pod.sh).

    python ws4/cloud/skill_env.py DATASET_DIR SIDE SKILL > skill.env

Checks: status "final"; train and held-out disjoint; no review.json reject in either; indices inside the dataset; one
task (read verbatim from meta/tasks.parquet). Prints the counts to stderr.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path


def skill_env(src: Path, side: str, skill: str) -> dict[str, str]:
    import pandas as pd

    split = json.loads((src / "ws4_split.json").read_text())
    if split["status"] != "final":
        raise SystemExit(f"split status is {split['status']}, not final")
    n = json.loads((src / "meta/info.json").read_text())["total_episodes"]
    review = json.loads((src / "review.json").read_text()) if (src / "review.json").exists() else {"episodes": {}}
    rejects = sorted(int(k) for k, v in review["episodes"].items() if v.get("status") == "reject")
    train = sorted(split["train"])
    held = sorted(e for v in split["heldout"].values() for e in v)
    if set(train) & set(held):
        raise SystemExit("train and held-out overlap")
    if (set(train) | set(held)) & set(rejects):
        raise SystemExit("a review.json reject is in train or held-out")
    if max(train + held) >= n:
        raise SystemExit(f"split names episode {max(train + held)}, dataset has {n}")
    tasks = list(pd.read_parquet(src / "meta/tasks.parquet").index)
    if len(tasks) != 1:
        raise SystemExit(f"expected one task, got {tasks}")
    print(f"# {skill}: train {len(train)}, held-out {len(held)} {split['heldout']}, rejects {rejects}, {n} episodes",
          file=sys.stderr)  # fmt: skip
    return {
        "SKILL": skill,
        "SIDE": side,
        "TASK": tasks[0],
        "N_EPISODES": str(n),
        "TRAIN_EPS": " ".join(map(str, train)),
        "EVAL_EPS": " ".join(map(str, held)),
        "STATS_EXCLUDE": " ".join(str(e) for e in range(n) if e not in set(train)),
        "EVAL_EXCLUDE": " ".join(str(e) for e in range(n) if e not in set(held)),
    }


def main(argv: list[str]) -> None:
    if len(argv) != 3 or argv[1] not in ("left", "right"):
        raise SystemExit("usage: skill_env.py DATASET_DIR left|right SKILL")
    for k, v in skill_env(Path(argv[0]), argv[1], argv[2]).items():
        print(f"{k}={shlex.quote(v)}")


if __name__ == "__main__":
    main(sys.argv[1:])
