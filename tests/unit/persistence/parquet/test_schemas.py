# ruff: noqa: ANN201
import polars as pl
import pytest

from evolucio.core.codes import DeathCauseCode
from evolucio.persistence.parquet.schemas import (
    BIRTHS_SCHEMA,
    DEATH_CAUSE_COLUMNS,
    DEATHS_SCHEMA,
    GENEALOGY_SCHEMA,
    METRICS_SCHEMA,
    PARQUET_DATASET_SCHEMA_VERSION,
    ParquetPersistencePolicy,
)


def test_frozen_v1_schemas():
    assert PARQUET_DATASET_SCHEMA_VERSION == 1
    assert tuple(f"deaths_{c.name.lower()}" for c in DeathCauseCode) == DEATH_CAUSE_COLUMNS
    assert BIRTHS_SCHEMA.names() == [
        "run_id",
        "birth_step",
        "child_agent_id",
        "parent_agent_id",
        "lineage_id",
        "generation",
        "child_genome_id",
        "parent_genome_id",
        "position_x",
        "position_y",
        "child_initial_energy",
        "parent_energy_before",
        "parent_energy_after",
        "mutation_selected_count",
        "mutation_effective_count",
        "mutation_selected_weight_count",
        "mutation_selected_bias_count",
        "mutation_sum_abs_delta",
        "mutation_max_abs_delta",
    ]
    assert DEATHS_SCHEMA.names() == [
        "run_id",
        "death_step",
        "agent_id",
        "parent_id",
        "lineage_id",
        "genome_id",
        "generation",
        "age",
        "energy",
        "position_x",
        "position_y",
        "cause_code",
        "phase_code",
    ]
    assert GENEALOGY_SCHEMA.names() == [
        "run_id",
        "birth_step",
        "parent_agent_id",
        "child_agent_id",
        "lineage_id",
        "generation",
        "parent_genome_id",
        "child_genome_id",
        "mutation_selected_count",
        "mutation_effective_count",
        "mutation_sum_abs_delta",
        "mutation_max_abs_delta",
    ]
    assert METRICS_SCHEMA.names() == [
        "run_id",
        "step",
        "alive_count",
        "birth_count",
        "birth_rejected_capacity",
        "death_count",
        *DEATH_CAUSE_COLUMNS,
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
        "death_energy_removed",
        "movement_success_count",
        "feeding_success_count",
        "reproduction_success_count",
        "active_lineage_count",
        "genetic_parameter_variance",
        "mean_environment",
    ]
    assert DEATHS_SCHEMA["phase_code"] == DEATHS_SCHEMA["cause_code"] == pl.Int32
    assert BIRTHS_SCHEMA["child_agent_id"] == pl.Int64
    assert ParquetPersistencePolicy().compression == "zstd"
    with pytest.raises(ValueError):
        ParquetPersistencePolicy(compression="bad")
