"""Canonical configuration freezing and hashing."""

import hashlib
import json
from dataclasses import dataclass

from .models import ExperimentConfig


def canonical_json_and_hash(value: object) -> tuple[str, str]:
    """Serialize a JSON-compatible value canonically and return its SHA-256."""
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenConfig:
    """A validated configuration and its process-independent identity."""

    config: ExperimentConfig
    canonical_json: str
    config_hash: str


def freeze_config(config: ExperimentConfig) -> FrozenConfig:
    """Return the canonical UTF-8 JSON representation and SHA-256 digest."""
    canonical, digest = canonical_json_and_hash(
        config.model_dump(mode="json", exclude_none=False, exclude_defaults=False)
    )
    return FrozenConfig(config=config, canonical_json=canonical, config_hash=digest)
