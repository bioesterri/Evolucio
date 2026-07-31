"""Canonical configuration freezing and hashing."""

import hashlib
import json
from dataclasses import dataclass

from .models import ExperimentConfig


@dataclass(frozen=True, slots=True)
class FrozenConfig:
    """A validated configuration and its process-independent identity."""

    config: ExperimentConfig
    canonical_json: str
    config_hash: str


def freeze_config(config: ExperimentConfig) -> FrozenConfig:
    """Return the canonical UTF-8 JSON representation and SHA-256 digest."""
    validated_config = ExperimentConfig.model_validate(
        config.model_dump(mode="python", exclude_none=False, exclude_defaults=False)
    )
    canonical = json.dumps(
        validated_config.model_dump(mode="json", exclude_none=False, exclude_defaults=False),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return FrozenConfig(config=validated_config, canonical_json=canonical, config_hash=digest)
