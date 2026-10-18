"""Generate the assembly task in a chosen placement order; no grammar extension."""
import argparse
from pathlib import Path

import yaml

from liminal.orchestration.component_assembly import COMPONENTS, assembly_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", nargs=3, choices=COMPONENTS, default=COMPONENTS)
    parser.add_argument("--output", type=Path, default=Path("configs/tasks/assemble_components.yaml"))
    args = parser.parse_args()
    try:
        task = assembly_task(args.order)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(task.raw, sort_keys=False))
    print(args.output)


if __name__ == "__main__":
    main()
