"""Versioned contract for pre-action viability resolution."""

import hashlib
import json

PRE_ACTION_VIABILITY_SCHEMA_NAME = "invalid_energy_age_preaction_death_v1"
PRE_ACTION_VIABILITY_SCHEMA_VERSION = 1


def pre_action_viability_schema_payload() -> dict[str, object]:
    """Return the complete JSON-compatible pre-action viability contract."""
    return {
        "name": PRE_ACTION_VIABILITY_SCHEMA_NAME,
        "version": PRE_ACTION_VIABILITY_SCHEMA_VERSION,
        "phase": "after_basal_metabolism_and_age_before_observations",
        "active_terminal_causes": ["INVALID_STATE", "ENERGY_DEPLETION", "MAX_AGE"],
        "cause_priority": ["INVALID_STATE", "ENERGY_DEPLETION", "MAX_AGE"],
        "energy_depletion": "alive and energy <= death_energy_threshold",
        "maximum_age": "alive and age >= maximum_age",
        "invalid_state": [
            "non_finite_energy",
            "negative_age",
            "position_out_of_bounds",
            "invalid_agent_id",
            "invalid_lineage_id",
            "invalid_genome_id",
            "negative_generation",
            "birth_step_negative_or_after_death_step",
        ],
        "finite_negative_energy": "energy_depletion_not_invalid_state",
        "direct_environmental_death": "none",
        "direct_competitive_death": "none",
        "death_record": "fixed_capacity_snapshot_before_slot_cleanup",
        "population_cleanup": "canonical_inactive_slot",
        "genome_cleanup": "all_parameter_leaves_zero",
        "spatial_update": "rebuild_world_occupancy_after_cleanup",
        "rng": "none",
    }


def pre_action_viability_schema_digest() -> str:
    """Return the SHA-256 digest of canonical schema JSON."""
    canonical = json.dumps(
        pre_action_viability_schema_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


PRE_ACTION_VIABILITY_SCHEMA_DIGEST = pre_action_viability_schema_digest()
