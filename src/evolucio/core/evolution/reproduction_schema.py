"""Versioned atomic asexual-reproduction contract."""

import hashlib
import json

REPRODUCTION_RESOLUTION_SCHEMA_NAME = "asexual_atomic_local_birth_fixed_slots_v1"
REPRODUCTION_RESOLUTION_SCHEMA_VERSION = 1


def reproduction_resolution_schema_payload() -> dict[str, object]:
    """Return the complete JSON-compatible reproduction contract."""
    return {
        "name": REPRODUCTION_RESOLUTION_SCHEMA_NAME,
        "version": REPRODUCTION_RESOLUTION_SCHEMA_VERSION,
        "mode": "asexual",
        "maximum_offspring_per_parent_per_step": 1,
        "energy_cost": "total_parent_debit_including_offspring_initial_energy",
        "inheritance": "exact_genome_copy_before_mutation",
        "birth_neighbours": ["north", "south", "east", "west"],
        "birth_cell": "in_bounds_and_empty_in_prebirth_occupancy_snapshot",
        "placement": "reproducible_local_choice_derived_from_agent_id",
        "spatial_conflict": "neutral_priority_derived_from_agent_id",
        "population": "fixed_capacity_masked_slots",
        "slot_assignment": "ascending_free_slots_after_identity_neutral_selection",
        "id_overflow": "atomic_global_rollback",
        "identifiers": "new_agent_id_and_genome_id_no_new_lineage_id",
        "lineage": "inherited_from_parent",
        "generation": "parent_plus_one",
        "initial_state": "configured_energy_zero_age_alive_current_birth_step",
        "mutation": "none",
    }


def reproduction_resolution_schema_digest() -> str:
    """Return the SHA-256 digest of canonical schema JSON."""
    canonical = json.dumps(
        reproduction_resolution_schema_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


REPRODUCTION_RESOLUTION_SCHEMA_DIGEST = reproduction_resolution_schema_digest()
