"""Public pre-action viability API."""

from .death import DeathRecordBatch, build_death_records
from .pre_action import PreActionViabilityResult, resolve_pre_action_viability
from .schema import (
    PRE_ACTION_VIABILITY_SCHEMA_DIGEST,
    PRE_ACTION_VIABILITY_SCHEMA_NAME,
    PRE_ACTION_VIABILITY_SCHEMA_VERSION,
    pre_action_viability_schema_digest,
    pre_action_viability_schema_payload,
)

__all__ = [
    "PRE_ACTION_VIABILITY_SCHEMA_DIGEST",
    "PRE_ACTION_VIABILITY_SCHEMA_NAME",
    "PRE_ACTION_VIABILITY_SCHEMA_VERSION",
    "DeathRecordBatch",
    "PreActionViabilityResult",
    "build_death_records",
    "pre_action_viability_schema_digest",
    "pre_action_viability_schema_payload",
    "resolve_pre_action_viability",
]
