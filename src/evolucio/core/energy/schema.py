"""Canonical schema for step energy accounting."""

import hashlib
import json

ENERGY_ACCOUNTING_SCHEMA_NAME = "preaction_basal_postaction_success_costs_v1"
ENERGY_ACCOUNTING_SCHEMA_VERSION = 1
FAILED_ACTION_COST_POLICY = "zero"
REPRODUCTION_COST_POLICY = "deferred_to_atomic_birth"
ENERGY_FLOOR_POLICY = "no_clipping_before_viability"


def energy_accounting_schema_payload() -> dict[str, object]:
    """Return the complete JSON-compatible energy-accounting contract."""
    return {
        "name": ENERGY_ACCOUNTING_SCHEMA_NAME,
        "version": ENERGY_ACCOUNTING_SCHEMA_VERSION,
        "phase_order": {
            "preaction": [
                "environment_update",
                "basal_metabolism_and_age",
                "preaction_viability",
                "observations_and_policy",
            ],
            "postaction": [
                "movement",
                "feeding",
                "successful_action_costs",
                "postaction_viability",
                "reproduction",
            ],
        },
        "age_increment": "once_per_live_agent_before_actions",
        "basal_cost": "once_per_live_agent_before_actions",
        "movement_cost": "MovementResolutionCode.MOVED_only",
        "feeding_cost": ["FeedingResolutionCode.FED_FULL", "FeedingResolutionCode.FED_PARTIAL"],
        "failed_action_cost_policy": FAILED_ACTION_COST_POLICY,
        "reproduction_cost_policy": REPRODUCTION_COST_POLICY,
        "energy_floor_policy": ENERGY_FLOOR_POLICY,
        "balance_formula": (
            "energy_after=energy_before-basal_cost+feeding_gain-movement_cost-feeding_cost"
        ),
        "rng": "none",
    }


def energy_accounting_schema_digest() -> str:
    """Return the SHA-256 digest of canonical schema JSON."""
    canonical = json.dumps(
        energy_accounting_schema_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


ENERGY_ACCOUNTING_SCHEMA_DIGEST = energy_accounting_schema_digest()
