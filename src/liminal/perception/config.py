"""Configuration loader for a persistent VLM and its predicate vocabulary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from liminal.perception.types import InspectionProfile
from liminal.perception.vlm import VisionModel, build_vision_model


@dataclass(frozen=True)
class PerceptionRuntime:
    model: VisionModel
    inspection: InspectionProfile
    confirmations: int = 2


def load_perception_config(path: str | Path) -> PerceptionRuntime:
    config_path = Path(path)
    try:
        data = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load perception config {config_path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("perception configuration must be a mapping")
    unknown = set(data) - {"model", "inspection"}
    if unknown:
        raise ValueError(f"unknown perception configuration fields: {sorted(unknown)}")

    model_data = data.get("model")
    if not isinstance(model_data, Mapping):
        raise ValueError("perception model configuration must be a mapping")
    inspection_data = data.get("inspection")
    if not isinstance(inspection_data, Mapping):
        raise ValueError("perception inspection configuration must be a mapping")
    inspection_values: dict[str, Any] = dict(inspection_data)
    confirmations = inspection_values.pop("confirmations", 2)
    if isinstance(confirmations, bool) or not isinstance(confirmations, int) or confirmations < 1:
        raise ValueError("inspection confirmations must be a positive integer")
    inspection_values["cameras"] = _strings(inspection_values.get("cameras", ["top"]), "inspection.cameras")
    groups = inspection_values.get("exclusive_groups", [])
    if not isinstance(groups, list):
        raise ValueError("inspection.exclusive_groups must be a list")
    inspection_values["exclusive_groups"] = tuple(
        _strings(group, f"inspection.exclusive_groups[{index}]") for index, group in enumerate(groups)
    )
    try:
        inspection = InspectionProfile(**inspection_values)
    except TypeError as exc:
        raise ValueError(f"invalid inspection configuration: {exc}") from exc
    return PerceptionRuntime(
        model=build_vision_model(model_data),
        inspection=inspection,
        confirmations=confirmations,
    )


def _strings(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{path} must be a non-empty list of strings")
    return tuple(value)
