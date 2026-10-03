"""Versioned schema for metrics, event buffers, and accumulators."""

import hashlib
import json

from evolucio.core.codes import DEATH_CAUSE_COUNT

from .events import DeathPhaseCode

METRICS_SCHEMA_NAME = "step_metrics_events_accumulators_v1"
METRICS_SCHEMA_VERSION = 1

_STEP_FIELDS = (
    "step",
    "alive_count",
    "birth_count",
    "death_count",
    "deaths_by_cause",
    "total_energy_alive",
    "mean_energy_alive",
    "mean_age_alive",
    "mean_age_at_death",
    "total_resources",
    "resource_consumed",
    "feeding_energy_gained",
    "basal_energy_cost",
    "movement_energy_cost",
    "feeding_energy_cost",
    "reproduction_energy_cost",
    "movement_success_count",
    "feeding_success_count",
    "reproduction_success_count",
    "active_lineage_count",
    "genetic_parameter_variance",
    "mean_environment",
)
_FINAL_STATE_FIELDS = (
    "alive_count",
    "total_energy_alive",
    "mean_energy_alive",
    "mean_age_alive",
    "total_resources",
    "active_lineage_count",
    "genetic_parameter_variance",
    "mean_environment",
)
_ACCUMULATOR_FIELDS = (
    "steps_accumulated",
    "births_total",
    "deaths_total",
    "deaths_by_cause_total",
    "resource_consumed_total",
    "feeding_energy_gained_total",
    "basal_energy_cost_total",
    "movement_energy_cost_total",
    "feeding_energy_cost_total",
    "reproduction_energy_cost_total",
    "movement_success_total",
    "feeding_success_total",
    "reproduction_success_total",
    "death_age_sum",
    "alive_count_sum",
    "energy_alive_sum",
    "count_overflow",
)


def metrics_schema_payload() -> dict[str, object]:
    """Return the complete JSON-compatible observation contract."""
    final_fields = set(_FINAL_STATE_FIELDS)
    return {
        "name": METRICS_SCHEMA_NAME,
        "version": METRICS_SCHEMA_VERSION,
        "step_metrics_fields": list(_STEP_FIELDS),
        "temporal_semantics": {
            field: "final_state" if field in final_fields else "within_step_flow"
            for field in _STEP_FIELDS
        },
        "death_phase_codes": [{"name": code.name, "value": int(code)} for code in DeathPhaseCode],
        "buffer_shapes": {
            "births": "population_capacity",
            "deaths": "population_capacity",
            "death_phase": "population_capacity",
            "deaths_by_cause": DEATH_CAUSE_COUNT,
        },
        "genetic_parameter_variance": (
            "mean(population_variance(parameter_across_alive_agents)) over all 375 parameters; "
            "zero when alive_count<=1"
        ),
        "active_lineage_count": "distinct lineage_id values among final live agents",
        "accumulator_fields": list(_ACCUMULATOR_FIELDS),
        "count_overflow_policy": "saturate_int32_and_set_count_overflow",
        "fitness": False,
        "persistence": False,
    }


def metrics_schema_digest() -> str:
    """Return the canonical SHA-256 digest of the metrics contract."""
    canonical = json.dumps(
        metrics_schema_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


METRICS_SCHEMA_DIGEST = metrics_schema_digest()
