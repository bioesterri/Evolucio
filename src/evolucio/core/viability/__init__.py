"""Public viability barriers and reproduction-gate API."""

from .death import DeathRecordBatch, build_death_records
from .post_action import PostActionViabilityResult, resolve_post_action_viability
from .pre_action import PreActionViabilityResult, resolve_pre_action_viability
from .reproduction_gate import (
    ReproductionGateCode,
    ReproductionGateResult,
    evaluate_reproduction_gate,
)
from .schema import (
    POST_ACTION_VIABILITY_SCHEMA_DIGEST,
    POST_ACTION_VIABILITY_SCHEMA_NAME,
    POST_ACTION_VIABILITY_SCHEMA_VERSION,
    PRE_ACTION_VIABILITY_SCHEMA_DIGEST,
    PRE_ACTION_VIABILITY_SCHEMA_NAME,
    PRE_ACTION_VIABILITY_SCHEMA_VERSION,
    post_action_viability_schema_digest,
    post_action_viability_schema_payload,
    pre_action_viability_schema_digest,
    pre_action_viability_schema_payload,
)

__all__ = [
    "POST_ACTION_VIABILITY_SCHEMA_DIGEST",
    "POST_ACTION_VIABILITY_SCHEMA_NAME",
    "POST_ACTION_VIABILITY_SCHEMA_VERSION",
    "PRE_ACTION_VIABILITY_SCHEMA_DIGEST",
    "PRE_ACTION_VIABILITY_SCHEMA_NAME",
    "PRE_ACTION_VIABILITY_SCHEMA_VERSION",
    "DeathRecordBatch",
    "PostActionViabilityResult",
    "PreActionViabilityResult",
    "ReproductionGateCode",
    "ReproductionGateResult",
    "build_death_records",
    "evaluate_reproduction_gate",
    "post_action_viability_schema_digest",
    "post_action_viability_schema_payload",
    "pre_action_viability_schema_digest",
    "pre_action_viability_schema_payload",
    "resolve_post_action_viability",
    "resolve_pre_action_viability",
]
