"""Canonical schema for simultaneous cardinal movement resolution."""

import hashlib
import json
from enum import IntEnum

from evolucio.core.codes import RngStreamCode

from .schema import ACTION_CONTRACT_SCHEMA_DIGEST

MOVEMENT_RESOLUTION_SCHEMA_NAME = "simultaneous_empty_snapshot_random_priority_v1"
MOVEMENT_RESOLUTION_SCHEMA_VERSION = 1
MOVEMENT_DESTINATION_POLICY = "empty_at_phase_start"
MOVEMENT_CONFLICT_POLICY = "lexicographic_uint32_priority_triplet"
MOVEMENT_UNRESOLVED_POLICY = "all_claimants_fail"
MOVEMENT_FAILED_ACTION = "stay"


class MovementResolutionCode(IntEnum):
    """Stable outcomes of the spatial movement phase."""

    NOT_MOVEMENT = 0
    MOVED = 1
    DESTINATION_OCCUPIED = 2
    CONFLICT_LOST = 3
    PRIORITY_COLLISION = 4
    INVALID_MOVEMENT_INPUT = 5


MOVEMENT_RESOLUTION_CODE_COUNT = len(MovementResolutionCode)


class MovementConflictPriorityStreamCode(IntEnum):
    """Independent components of an identity-bound conflict priority."""

    PRIORITY_0 = 0
    PRIORITY_1 = 1
    PRIORITY_2 = 2


_RESOLUTION_PRECEDENCE = (
    MovementResolutionCode.NOT_MOVEMENT,
    MovementResolutionCode.INVALID_MOVEMENT_INPUT,
    MovementResolutionCode.DESTINATION_OCCUPIED,
    MovementResolutionCode.PRIORITY_COLLISION,
    MovementResolutionCode.MOVED,
    MovementResolutionCode.CONFLICT_LOST,
)


def movement_resolution_schema_payload() -> dict[str, object]:
    """Return the complete JSON-compatible spatial resolution contract."""
    return {
        "name": MOVEMENT_RESOLUTION_SCHEMA_NAME,
        "version": MOVEMENT_RESOLUTION_SCHEMA_VERSION,
        "action_contract_digest": ACTION_CONTRACT_SCHEMA_DIGEST,
        "resolution_codes": [
            {"name": code.name, "value": int(code)} for code in MovementResolutionCode
        ],
        "resolution_precedence": [code.name for code in _RESOLUTION_PRECEDENCE],
        "temporal_policy": "simultaneous_immutable_occupancy_snapshot",
        "destination_policy": MOVEMENT_DESTINATION_POLICY,
        "maximum_entries_per_initially_empty_cell": 1,
        "preexisting_colocation": "preserved_without_expulsion",
        "swaps": "blocked",
        "cycles": "blocked",
        "vacancy_chains": "blocked",
        "rng_domain": RngStreamCode.MOVEMENT_CONFLICT.name,
        "identity_derivation": "agent_id",
        "priority_substreams": [
            {"name": code.name, "value": int(code)} for code in MovementConflictPriorityStreamCode
        ],
        "priority_components": ["uint32", "uint32", "uint32"],
        "priority_comparison": "lexicographic_minimum",
        "conflict_policy": MOVEMENT_CONFLICT_POLICY,
        "unresolved_policy": MOVEMENT_UNRESOLVED_POLICY,
        "failed_action": MOVEMENT_FAILED_ACTION,
        "applies_costs": False,
        "occupancy_update": "rebuild_world_occupancy",
    }


def movement_resolution_schema_digest() -> str:
    """Return the canonical SHA-256 digest of the movement contract."""
    canonical = json.dumps(
        movement_resolution_schema_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


MOVEMENT_RESOLUTION_SCHEMA_DIGEST = movement_resolution_schema_digest()
