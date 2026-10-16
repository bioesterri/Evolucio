"""Versioned contract for one complete ecological-evolutionary step."""

import hashlib
import json

from evolucio.core.actions import (
    ACTION_CONTRACT_SCHEMA_DIGEST,
    FEEDING_RESOLUTION_SCHEMA_DIGEST,
    MOVEMENT_RESOLUTION_SCHEMA_DIGEST,
)
from evolucio.core.diagnostics import INVARIANT_SCHEMA_DIGEST
from evolucio.core.energy import ENERGY_ACCOUNTING_SCHEMA_DIGEST
from evolucio.core.evolution import (
    GENEALOGY_SCHEMA_DIGEST,
    GENOME_MUTATION_SCHEMA_DIGEST,
    REPRODUCTION_RESOLUTION_SCHEMA_DIGEST,
)
from evolucio.core.metrics import METRICS_SCHEMA_DIGEST
from evolucio.core.observations import OBSERVATION_SCHEMA_DIGEST
from evolucio.core.policy import ACTION_SELECTION_SCHEMA_DIGEST, POLICY_SCHEMA_DIGEST
from evolucio.core.viability import (
    POST_ACTION_VIABILITY_SCHEMA_DIGEST,
    PRE_ACTION_VIABILITY_SCHEMA_DIGEST,
)

SIMULATION_STEP_SCHEMA_NAME = "full_ecological_evolution_step_v1"
SIMULATION_STEP_SCHEMA_VERSION = 1

SIMULATION_STEP_PHASES = (
    "advance_rng",
    "update_world_and_environment",
    "basal_metabolism_and_age",
    "pre_action_viability",
    "build_observations",
    "policy_inference",
    "deterministic_action_selection",
    "validate_and_route_actions",
    "resolve_movement",
    "resolve_feeding",
    "apply_action_costs",
    "post_action_viability",
    "evaluate_reproduction_gate",
    "resolve_asexual_reproduction",
    "mutate_newborn_genomes",
    "build_birth_events",
    "combine_death_events",
    "build_step_event_buffer",
    "build_final_ecological_state",
    "compute_step_metrics",
    "update_metrics_accumulator",
    "increment_step",
    "install_next_rng",
    "return_result",
)


def simulation_step_schema_payload() -> dict[str, object]:
    """Return the canonical, JSON-compatible complete-step contract."""
    return {
        "name": SIMULATION_STEP_SCHEMA_NAME,
        "version": SIMULATION_STEP_SCHEMA_VERSION,
        "phases": list(SIMULATION_STEP_PHASES),
        "step_semantics": "state.step_is_the_step_about_to_execute",
        "step_increment": "exactly_once_at_end",
        "rng_advance": "exactly_once_unconditionally",
        "rng_streams": [
            "MOVEMENT_CONFLICT",
            "REPRODUCTION_CONFLICT",
            "BIRTH_PLACEMENT",
            "GENOME_MUTATION",
        ],
        "preaction_viability_before_observations": True,
        "postaction_viability_before_reproduction": True,
        "birth_after_action_costs": True,
        "mutation_after_genome_copy": True,
        "newborns_act_next_step": True,
        "events_before_metrics": True,
        "metrics_describe_final_state": True,
        "accumulator_separate_from_state": True,
        "persistence": False,
        "contract_digests": {
            "action": ACTION_CONTRACT_SCHEMA_DIGEST,
            "action_selection": ACTION_SELECTION_SCHEMA_DIGEST,
            "energy": ENERGY_ACCOUNTING_SCHEMA_DIGEST,
            "feeding": FEEDING_RESOLUTION_SCHEMA_DIGEST,
            "genealogy": GENEALOGY_SCHEMA_DIGEST,
            "invariants": INVARIANT_SCHEMA_DIGEST,
            "metrics": METRICS_SCHEMA_DIGEST,
            "movement": MOVEMENT_RESOLUTION_SCHEMA_DIGEST,
            "mutation": GENOME_MUTATION_SCHEMA_DIGEST,
            "observations": OBSERVATION_SCHEMA_DIGEST,
            "policy": POLICY_SCHEMA_DIGEST,
            "post_action_viability": POST_ACTION_VIABILITY_SCHEMA_DIGEST,
            "pre_action_viability": PRE_ACTION_VIABILITY_SCHEMA_DIGEST,
            "reproduction": REPRODUCTION_RESOLUTION_SCHEMA_DIGEST,
        },
    }


def simulation_step_schema_digest() -> str:
    """Return the canonical SHA-256 digest of the complete-step schema."""
    canonical = json.dumps(
        simulation_step_schema_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


SIMULATION_STEP_SCHEMA_DIGEST = simulation_step_schema_digest()

RUN_CHUNK_SCHEMA_NAME = "compiled_lax_scan_simulation_chunk_v1"
RUN_CHUNK_SCHEMA_VERSION = 1
RUN_CHUNK_SCAN_KIND = "lax_scan"
RUN_CHUNK_UNROLL = 1
RUN_CHUNK_EARLY_STOP = False
RUN_CHUNK_OUTPUT_POLICY = "metrics_and_events_only"


def run_chunk_schema_payload() -> dict[str, object]:
    """Return the canonical compiled-chunk execution contract."""
    return {
        "name": RUN_CHUNK_SCHEMA_NAME,
        "version": RUN_CHUNK_SCHEMA_VERSION,
        "simulation_step_schema_digest": SIMULATION_STEP_SCHEMA_DIGEST,
        "iterated_unit": "simulation_step",
        "carry": ["SimulationState", "MetricsAccumulator"],
        "outputs": ["StepMetrics", "StepEventBuffer"],
        "xs": None,
        "length": "fixed_chunk_steps",
        "temporal_order": "forward",
        "scan_kind": RUN_CHUNK_SCAN_KIND,
        "unroll": RUN_CHUNK_UNROLL,
        "early_stop": RUN_CHUNK_EARLY_STOP,
        "output_policy": RUN_CHUNK_OUTPUT_POLICY,
        "state_trajectory": False,
        "fast_path_diagnostics": False,
        "compilation": "equinox_filter_jit_over_jax_lax_scan",
        "chunk_steps": "static_positive_python_int_execution_argument",
        "chunk_steps_change_may_recompile": True,
        "chunk_steps_in_scientific_config_hash": False,
    }


def run_chunk_schema_digest() -> str:
    """Return the canonical SHA-256 digest of the chunk schema."""
    canonical = json.dumps(
        run_chunk_schema_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


RUN_CHUNK_SCHEMA_DIGEST = run_chunk_schema_digest()
