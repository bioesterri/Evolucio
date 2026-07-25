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
    canonical = json.dumps(
        config.model_dump(mode="json", exclude_none=False, exclude_defaults=False),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return FrozenConfig(config=config, canonical_json=canonical, config_hash=digest)
