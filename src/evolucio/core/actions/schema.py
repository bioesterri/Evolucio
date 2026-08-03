"""Canonical versioned schema for local action validation."""

import hashlib
import json

from evolucio.core.codes import ACTION_COUNT, ActionCode

from .contract import ACTION_DELTAS, ActionValidationCode

ACTION_CONTRACT_SCHEMA_NAME = "discrete_actions_local_validation_stay_fallback_v1"
ACTION_CONTRACT_SCHEMA_VERSION = 1
ACTION_INVALID_FALLBACK = "stay"
ACTION_VALIDATION_SCOPE = "local_preliminary"

_VALIDATION_PRECEDENCE = (
    ActionValidationCode.INACTIVE_SLOT,
    ActionValidationCode.INVALID_ACTION_CODE,
    ActionValidationCode.INVALID_ACTOR_POSITION,
    ActionValidationCode.MOVE_OUT_OF_BOUNDS,
    ActionValidationCode.EAT_NO_RESOURCE,
    ActionValidationCode.ACCEPTED,
)


def action_contract_schema_payload() -> dict[str, object]:
    """Return the complete JSON-compatible action-routing contract."""
    return {
        "name": ACTION_CONTRACT_SCHEMA_NAME,
        "version": ACTION_CONTRACT_SCHEMA_VERSION,
        "validation_scope": ACTION_VALIDATION_SCOPE,
        "action_codes": [{"name": code.name, "value": int(code)} for code in ActionCode],
        "action_count": ACTION_COUNT,
        "action_deltas_xy": [list(delta) for delta in ACTION_DELTAS],
        "coordinate_orientation": "position[...,0]=x;position[...,1]=y;north_is_negative_y",
        "boundary_mode": "closed_non_wrapping",
        "validation_codes": [
            {"name": code.name, "value": int(code)} for code in ActionValidationCode
        ],
        "validation_precedence": [code.name for code in _VALIDATION_PRECEDENCE],
        "invalid_fallback": ACTION_INVALID_FALLBACK,
        "explicit_stay": "accepted_when_active_code_and_position_are_valid",
        "fallback_stay": "rejected_proposal_is_preserved_with_exact_validation_code",
        "eat_condition": "resource_at_actor_position>0",
        "consults_occupancy": False,
        "reproduction": "deferred_after_local_actor_validation",
        "applies_costs": False,
        "uses_randomness": False,
        "action_levels": {
            "proposed": "neural_decision",
            "routed": "locally_validated_intention_not_yet_executed",
            "executed": "future_specialized_resolution_result",
        },
    }


def action_contract_schema_digest() -> str:
    """Return the canonical SHA-256 digest of the action contract."""
    canonical = json.dumps(
        action_contract_schema_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


ACTION_CONTRACT_SCHEMA_DIGEST = action_contract_schema_digest()
