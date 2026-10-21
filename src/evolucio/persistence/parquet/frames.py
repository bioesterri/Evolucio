"""Pure columnar projections of compact host telemetry."""

from collections.abc import Sequence
from dataclasses import fields
from typing import Any

import polars as pl

from evolucio.host import BirthEventDTO, DeathEventDTO, StepMetricsDTO
from evolucio.persistence.parquet.schemas import DEFAULT_PARQUET_POLICY

from .schemas import (
    BIRTHS_SCHEMA,
    DEATH_CAUSE_COLUMNS,
    DEATHS_SCHEMA,
    GENEALOGY_COLUMNS,
    METRICS_SCHEMA,
    ParquetPersistencePolicy,
)


def _frame(
    run_id: str, rows: Sequence[Any], schema: pl.Schema, keys: tuple[str, ...]
) -> pl.DataFrame:
    order = [tuple(getattr(row, key) for key in keys) for row in rows]
    if order != sorted(order) or len(order) != len(set(order)):
        raise ValueError("telemetry is not in unique canonical order")
    columns: dict[str, Any] = {"run_id": [run_id] * len(rows)}
    if rows:
        for field in fields(rows[0]):
            if field.name == "deaths_by_cause":
                if any(len(row.deaths_by_cause) != len(DEATH_CAUSE_COLUMNS) for row in rows):
                    raise ValueError("invalid death cause vector")
                for i, name in enumerate(DEATH_CAUSE_COLUMNS):
                    columns[name] = [row.deaths_by_cause[i] for row in rows]
            else:
                columns[field.name] = [getattr(row, field.name) for row in rows]
    else:
        columns = {name: [] for name in schema}
    frame = pl.DataFrame(columns, schema=schema, strict=True)
    for name in schema:
        if (
            name
            in (
                "child_agent_id",
                "parent_agent_id",
                "agent_id",
                "lineage_id",
                "genome_id",
                "child_genome_id",
                "parent_genome_id",
            )
            and frame.filter(pl.col(name) < 0).height
        ):
            raise ValueError("invalid sentinel biological identity")
    if (
        "cause_code" in schema
        and frame.filter(
            ~pl.col("cause_code").is_in([1, 2, 3, 4, 5]) | ~pl.col("phase_code").is_in([1, 2])
        ).height
    ):
        raise ValueError("invalid death cause or phase")
    return frame


def metrics_frame(
    run_id: str,
    rows: Sequence[StepMetricsDTO],
    policy: ParquetPersistencePolicy = DEFAULT_PARQUET_POLICY,
) -> pl.DataFrame:
    frame = _frame(run_id, rows, METRICS_SCHEMA, ("step",))
    return frame.filter(pl.col("step") % policy.metrics_interval == 0)


def births_frame(run_id: str, rows: Sequence[BirthEventDTO]) -> pl.DataFrame:
    frame = _frame(run_id, rows, BIRTHS_SCHEMA, ("birth_step", "child_agent_id"))
    if frame["child_agent_id"].n_unique() != frame.height:
        raise ValueError("duplicate birth child")
    return frame


def deaths_frame(run_id: str, rows: Sequence[DeathEventDTO]) -> pl.DataFrame:
    return _frame(run_id, rows, DEATHS_SCHEMA, ("death_step", "agent_id"))


def genealogy_frame(run_id: str, rows: Sequence[BirthEventDTO]) -> pl.DataFrame:
    return births_frame(run_id, rows).select(GENEALOGY_COLUMNS)
