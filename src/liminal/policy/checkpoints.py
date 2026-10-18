"""Configuration-only selection of immutable, locally staged policy bundles."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Checkpoint:
    name: str
    path: str
    base: str
    sha256: str
    action_dim: int
    camera_roles: tuple[str, ...]
    targets: tuple[str, ...]
    embodiment: str
    joint_order: tuple[str, ...]
    state_units: str
    action_semantics: str
    obs_history: int = 8
    chunk_size: int = 42
    control_hz: float = 30.0
    normalization_profile: str = "released"

    @classmethod
    def parse(cls, name: str, values: dict[str, Any]) -> Checkpoint:
        try:
            return cls(name=name, **{**values, **{key: tuple(values[key])
                                                for key in ("camera_roles", "joint_order", "targets")}})
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{name}: incomplete checkpoint contract: {exc}") from exc

    def validate(self, cfg: Any, base: str, target: str) -> None:
        if not base or self.base != base:
            raise ValueError(f"{self.name}: checkpoint must derive from the shared midtraining base")
        if target not in self.targets:
            raise ValueError(f"{self.name}: checkpoint is not approved for target {target!r}")
        if len(self.joint_order) != self.action_dim or len(set(self.joint_order)) != self.action_dim:
            raise ValueError(f"{self.name}: joint order must name every action dimension uniquely")
        for key in ("action_dim", "camera_roles", "obs_history", "chunk_size", "control_hz",
                    "embodiment", "joint_order", "state_units", "action_semantics"):
            if getattr(self, key) != getattr(cfg, key):
                raise ValueError(f"{self.name}: incompatible {key}")
        if self.normalization_profile not in ("released", "hf_space_native", "dataset_native"):
            raise ValueError(f"{self.name}: unknown normalization profile")

    def identity(self) -> dict[str, str]:
        """Hash all bundle files, so paths cannot silently stand in for weight identities."""
        root = Path(self.path).expanduser().resolve(strict=True)
        if not root.is_dir() or not (root / "config.json").is_file():
            raise ValueError(f"{root}: expected a staged LeRobot checkpoint directory")
        paths = sorted(p for p in root.rglob("*") if p.is_file())
        if not any(p.suffix == ".safetensors" for p in paths):
            raise ValueError(f"{root}: no saved weights/processors")
        digest = hashlib.sha256()
        for path in paths:
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        provenance = root / "liminal_provenance.json"
        if not provenance.is_file():
            raise ValueError(f"{root}: missing midtraining provenance")
        manifest = json.loads(provenance.read_text())
        for key in ("base", "action_dim", "camera_roles", "targets", "embodiment", "joint_order",
                    "state_units", "action_semantics", "obs_history", "chunk_size", "control_hz",
                    "normalization_profile"):
            expected = getattr(self, key)
            if isinstance(expected, tuple):
                expected = list(expected)
            if manifest.get(key) != expected:
                raise ValueError(f"{root}: provenance mismatch for {key}")
        if digest.hexdigest() != self.sha256:
            raise ValueError(f"{root}: checkpoint digest mismatch")
        return {"name": self.name, "path": str(root), "base": self.base, "sha256": digest.hexdigest()}
