"""Versioned genealogy and birth-event semantic contract."""

import hashlib
import json

GENEALOGY_SCHEMA_NAME = "founder_lineage_parent_edge_birth_event_v1"
GENEALOGY_SCHEMA_VERSION = 1


def genealogy_schema_payload() -> dict[str, object]:
    """Return the canonical JSON-compatible genealogy contract."""
    return {
        "schema_name": GENEALOGY_SCHEMA_NAME,
        "schema_version": GENEALOGY_SCHEMA_VERSION,
        "identifiers": {
            "agent_id": "unique individual identity",
            "parent_id": "direct genealogical edge to parent agent_id",
            "lineage_id": "founder ancestry identity",
            "genome_id": "unique individual genome identity",
            "core_event_id": False,
        },
        "founders": {
            "parent_id": "NULL_ID",
            "generation": 0,
            "lineage": "one newly allocated lineage_id per founder",
        },
        "descendants": {
            "lineage": "inherit parent.lineage_id without allocation",
            "generation": "parent.generation + 1",
            "birth_step": "current step",
            "genomes": "parent_genome_id to new child_genome_id",
        },
        "birth_event_batch": {
            "shape": "fixed population capacity C aligned with newborn slots",
            "active_row_limit": "at most max_births_per_step born rows",
            "buffer_boundary": "slot-aligned core trace, not the compact PR-25 chunk buffer",
            "fields": [
                "born",
                "birth_step",
                "child_agent_id",
                "parent_agent_id",
                "lineage_id",
                "generation",
                "child_genome_id",
                "parent_genome_id",
                "birth_position",
                "child_initial_energy",
                "parent_energy_before",
                "parent_energy_after",
                "mutation_selected_count",
                "mutation_effective_count",
                "mutation_selected_weight_count",
                "mutation_selected_bias_count",
                "mutation_sum_abs_delta",
                "mutation_max_abs_delta",
            ],
            "mutation_summary": "copied from GenomeMutationResult",
            "inactive_rows": "canonical null, zero, and [-1,-1] values",
        },
        "population_state": {"offspring_count": False},
    }


def genealogy_schema_digest() -> str:
    """Return the SHA-256 digest of canonical compact JSON."""
    canonical = json.dumps(
        genealogy_schema_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


GENEALOGY_SCHEMA_DIGEST = genealogy_schema_digest()
