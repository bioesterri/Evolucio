"""Canonical configuration freezing and hashing."""

import hashlib
import json
from dataclasses import dataclass
from typing import cast

from pydantic import BaseModel

from .models import ExperimentConfig


@dataclass(frozen=True, slots=True)
class FrozenConfig:
    """A validated configuration and its process-independent identity."""

    config: ExperimentConfig
    canonical_json: str
    config_hash: str


def _validation_data(value: object) -> object:
    """Convert models to raw data without discarding unchecked extra fields."""
    if isinstance(value, BaseModel):
        model_values = cast(dict[str, object], vars(value))
        return {key: _validation_data(item) for key, item in model_values.items()}
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        return {key: _validation_data(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        return tuple(_validation_data(item) for item in sequence)
    return value


def freeze_config(config: ExperimentConfig) -> FrozenConfig:
    """Return the canonical UTF-8 JSON representation and SHA-256 digest."""
    validated_config = ExperimentConfig.model_validate(_validation_data(config))
    canonical = json.dumps(
        validated_config.model_dump(mode="json", exclude_none=False, exclude_defaults=False),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return FrozenConfig(config=validated_config, canonical_json=canonical, config_hash=digest)
