"""Versioned semantic schema for invariant diagnostics."""

import hashlib
import json

from .codes import StateInvariantCode, TransitionInvariantCode

INVARIANT_SCHEMA_NAME = "core_state_transition_invariants_v2"
INVARIANT_SCHEMA_VERSION = 2


def invariant_schema_payload() -> dict[str, object]:
    """Return the complete JSON-compatible invariant contract."""
    return {
        "name": INVARIANT_SCHEMA_NAME,
        "version": INVARIANT_SCHEMA_VERSION,
        "state_codes": {code.name: int(code) for code in StateInvariantCode},
        "transition_codes": {code.name: int(code) for code in TransitionInvariantCode},
        "levels": {
            "structural": (
                "internal consistency usable during initialization and intermediate phases"
            ),
            "final": "structural consistency plus post-mortality viability",
            "transition": (
                "consistency of before/after states, events, metrics, counters, step, and RNG"
            ),
        },
        "inactive_slots": {
            "ids": -1,
            "generation_birth_step_age": 0,
            "position": [-1, -1],
            "energy": 0,
            "genome_leaves": 0,
        },
        "checks": {
            "spatial": (
                "active positions are in bounds; occupancy is nonnegative and equals recomputation"
            ),
            "resources": (
                "resources are finite and within inclusive capacity; environment is finite"
            ),
            "identity": (
                "active agent and genome IDs are currently unique and counters exceed active IDs"
            ),
            "genealogy": (
                "descendant parents precede child IDs; active founder lineages are unique"
            ),
            "birth_step": "active birth steps are nonnegative and do not exceed the state step",
            "historical_identity_limit": (
                "current state cannot prove that an ID was never reused historically; "
                "persistent records must establish that property"
            ),
            "final_viability": (
                "alive energy exceeds death threshold and alive age is below maximum age"
            ),
            "population_balance": (
                "alive_after equals alive_before plus recorded births minus recorded deaths"
            ),
            "events_metrics": (
                "birth records match newborn slots, death identities match removed agents, "
                "terminal causes and structural step metrics agree"
            ),
            "step": "step advances exactly once",
            "rng": "persistent RNG equals exactly one advance_rng application",
        },
        "policy": "detect, never repair",
    }


def invariant_schema_digest() -> str:
    """Hash the canonical invariant payload with SHA-256."""
    canonical = json.dumps(
        invariant_schema_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


INVARIANT_SCHEMA_DIGEST = invariant_schema_digest()
