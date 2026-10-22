"""Versioned analytical schemas, independent of device libraries."""

from dataclasses import dataclass, fields
from typing import Literal

import polars as pl

from evolucio.host import BirthEventDTO, DeathEventDTO, StepMetricsDTO

PARQUET_DATASET_SCHEMA_NAME = "run_telemetry_parquet_v1"
PARQUET_DATASET_SCHEMA_VERSION = 1
DEATH_CAUSE_COLUMNS = (
    "deaths_none",
    "deaths_energy_depletion",
    "deaths_max_age",
    "deaths_environmental_stress",
    "deaths_competitive_exclusion",
    "deaths_invalid_state",
)


type ParquetCompression = Literal["zstd", "lz4", "uncompressed", "snappy", "gzip", "brotli"]


@dataclass(frozen=True, slots=True)
class ParquetPersistencePolicy:
    metrics_interval: int = 1
    persist_births: bool = True
    persist_deaths: bool = True
    persist_genealogy: bool = True
    compression: ParquetCompression = "zstd"

    def __post_init__(self) -> None:
        if type(self.metrics_interval) is not int or self.metrics_interval <= 0:
            raise ValueError("metrics_interval must be a positive integer")
        if any(
            value is not True
            for value in (self.persist_births, self.persist_deaths, self.persist_genealogy)
        ):
            raise ValueError("births, deaths and genealogy must always be persisted")
        if self.compression not in ("zstd", "lz4", "uncompressed", "snappy", "gzip", "brotli"):
            raise ValueError("unsupported Parquet compression")


def _schema(dto: type) -> pl.Schema:
    columns: dict[str, type[pl.DataType]] = {"run_id": pl.String}
    for field in fields(dto):
        if field.name == "deaths_by_cause":
            columns.update(dict.fromkeys(DEATH_CAUSE_COLUMNS, pl.Int64))
        else:
            columns[field.name] = (
                pl.Float32
                if field.type == "float"
                else pl.Int32
                if field.name.endswith("_code")
                else pl.Int64
            )
    return pl.Schema(columns)


METRICS_SCHEMA = _schema(StepMetricsDTO)
BIRTHS_SCHEMA = _schema(BirthEventDTO)
DEATHS_SCHEMA = _schema(DeathEventDTO)
GENEALOGY_COLUMNS = (
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
)
GENEALOGY_SCHEMA = pl.Schema({name: BIRTHS_SCHEMA[name] for name in GENEALOGY_COLUMNS})
SCHEMAS = {
    "metrics": METRICS_SCHEMA,
    "births": BIRTHS_SCHEMA,
    "deaths": DEATHS_SCHEMA,
    "genealogy": GENEALOGY_SCHEMA,
}
STEP_COLUMNS = {
    "metrics": "step",
    "births": "birth_step",
    "deaths": "death_step",
    "genealogy": "birth_step",
}

DEFAULT_PARQUET_POLICY = ParquetPersistencePolicy()
