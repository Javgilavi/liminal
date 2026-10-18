"""Verify staged bundle provenance and hashes without loading a model or touching motors."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml

from liminal.app import build_policy
from liminal.policy.checkpoints import Checkpoint


def bundle_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", nargs="?", default="configs/policy/showcase_registry.yaml")
    parser.add_argument("--hash-bundle", type=Path, help="print a staged bundle's digest for deployment configuration")
    args = parser.parse_args()
    if args.hash_bundle:
        if not (args.hash_bundle / "liminal_provenance.json").is_file():
            parser.error("bundle requires a liminal_provenance.json supplied by its training/export pipeline")
        print(bundle_hash(args.hash_bundle))
        return
    backend = build_policy("flux", yaml.safe_load(Path(args.policy).read_text()))
    cfg = backend.cfg
    for skill, name in cfg.skill_checkpoints.items():
        entry = Checkpoint.parse(name, cfg.checkpoints[name])
        for target in entry.targets:
            entry.validate(cfg, cfg.midtraining_base, target)
        print(json.dumps({"skill": skill, **entry.identity()}))


if __name__ == "__main__":
    main()
