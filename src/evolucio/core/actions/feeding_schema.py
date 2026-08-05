"""Canonical schema for local simultaneous feeding resolution."""

import hashlib
import json
from enum import IntEnum

FEEDING_RESOLUTION_SCHEMA_NAME = "local_proportional_feasible_demand_resource_transfer_v1"
FEEDING_RESOLUTION_SCHEMA_VERSION = 1
FEEDING_ALLOCATION_POLICY = "proportional_feasible_demand"
FEEDING_TRANSFER_POLICY = "resource_to_agent_only"
FEEDING_FAILED_ACTION = "stay"
FEEDING_CONFLICT_RANDOMNESS = "none"


class FeedingResolutionCode(IntEnum):
    """Stable outcomes of the feeding phase."""

    NOT_FEEDING = 0
    FED_FULL = 1
    FED_PARTIAL = 2
    NO_RESOURCE = 3
    NO_ENERGY_CAPACITY = 4
    INVALID_FEEDING_INPUT = 5


FEEDING_RESOLUTION_CODE_COUNT = 6
_RESOLUTION_PRECEDENCE = (
    FeedingResolutionCode.NOT_FEEDING,
    FeedingResolutionCode.INVALID_FEEDING_INPUT,
    FeedingResolutionCode.NO_RESOURCE,
    FeedingResolutionCode.NO_ENERGY_CAPACITY,
    FeedingResolutionCode.FED_PARTIAL,
    FeedingResolutionCode.FED_FULL,
)


def feeding_resolution_schema_payload() -> dict[str, object]:
    """Return the complete JSON-compatible feeding contract."""
    return {
        "name": FEEDING_RESOLUTION_SCHEMA_NAME,
        "version": FEEDING_RESOLUTION_SCHEMA_VERSION,
        "demand_formula": (
            "min(feeding_max_resource_intake, "
            "max(maximum_energy-energy, 0)/energy_gain_per_resource)"
        ),
        "allocation_policy": FEEDING_ALLOCATION_POLICY,
        "conversion_formula": "energy_gain=resource_consumed*energy_gain_per_resource",
        "energy_saturation": "min(maximum_energy, energy+theoretical_gain)",
        "resolution_codes": [
            {"name": code.name, "value": int(code)} for code in FeedingResolutionCode
        ],
        "resolution_precedence": [code.name for code in _RESOLUTION_PRECEDENCE],
        "failed_action": FEEDING_FAILED_ACTION,
        "conflict_randomness": FEEDING_CONFLICT_RANDOMNESS,
        "applies_costs": False,
        "transfer_policy": FEEDING_TRANSFER_POLICY,
        "agent_to_agent_transfer": False,
    }


def feeding_resolution_schema_digest() -> str:
    """Return the canonical SHA-256 digest of the feeding contract."""
    canonical = json.dumps(
        feeding_resolution_schema_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


FEEDING_RESOLUTION_SCHEMA_DIGEST = feeding_resolution_schema_digest()
