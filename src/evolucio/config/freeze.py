"""Canonical configuration freezing and hashing."""

import hashlib
import json
from dataclasses import dataclass
from typing import cast

from .models import ExperimentConfig


def _normalize_negative_zero(value: object) -> object:
    """Normalize signed zero recursively without changing other validated values."""
    if isinstance(value, float) and value == 0.0:
        return 0.0
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        return {key: _normalize_negative_zero(item) for key, item in mapping.items()}
    if isinstance(value, tuple):
        items = cast(tuple[object, ...], value)
        return tuple(_normalize_negative_zero(item) for item in items)
    return value


@dataclass(frozen=True, slots=True)
class FrozenConfig:
    """A validated configuration and its process-independent identity."""

    config: ExperimentConfig
    canonical_json: str
    config_hash: str


def freeze_config(config: ExperimentConfig) -> FrozenConfig:
    """Return the canonical UTF-8 JSON representation and SHA-256 digest."""
    normalized_data = _normalize_negative_zero(config.model_dump(mode="python"))
    validated_config = ExperimentConfig.model_validate(normalized_data)
    canonical = json.dumps(
        validated_config.model_dump(mode="json", exclude_none=False, exclude_defaults=False),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return FrozenConfig(config=validated_config, canonical_json=canonical, config_hash=digest)
