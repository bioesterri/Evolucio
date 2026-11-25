"""Structural checks over tabular data and explicit committed intervals."""

import polars as pl

from evolucio.analysis.summaries import (
    AGENT_SCHEMA,
    AGENTS_PATH,
    FOUNDER_SCHEMA,
    FOUNDERS_PATH,
    LINEAGE_SCHEMA,
    LINEAGES_PATH,
)
from evolucio.persistence.loaders.errors import DataConsistencyError, DataSchemaError
from evolucio.persistence.loaders.run_data import RunData
from evolucio.persistence.local.artifacts import artifact_path
from evolucio.persistence.parquet.schemas import SCHEMAS

from .scientific import row_count

_ANALYSIS: dict[str, tuple[pl.Schema, frozenset[str]]] = {
    FOUNDERS_PATH: (FOUNDER_SCHEMA, frozenset()),
    AGENTS_PATH: (
        AGENT_SCHEMA,
        frozenset(("parent_agent_id", "death_step", "death_cause_code", "death_phase_code")),
    ),
    LINEAGES_PATH: (LINEAGE_SCHEMA, frozenset(("last_birth_step", "extinction_step"))),
}


def check_analysis(run: RunData, paths: tuple[str, ...]) -> None:
    for record in run.manifest.artifacts:
        relative = record.relative_path
        if relative not in paths:
            continue
        schema, nullable = _ANALYSIS[relative]
        frame = run.scan_analysis(relative)
        if frame.collect_schema() != schema or row_count(
            frame.filter(pl.any_horizontal(pl.col(set(schema.names()) - nullable).is_null()))
        ):
            raise DataSchemaError(f"{relative}: invalid schema or required null")
        if "run_id" in schema and row_count(frame.filter(pl.col("run_id") != run.manifest.run_id)):
            raise DataSchemaError(f"{relative}: wrong run_id")


def check_chunks(run: RunData, *, scientific: bool) -> None:
    index = run.parquet
    assert index is not None
    start = 0 if run.recovery is not None else run.manifest.initial_step
    for name, dataset in index.datasets.items():
        if [part.chunk_index for part in dataset.parts] != sorted(
            part.chunk_index for part in dataset.parts
        ):
            raise DataSchemaError(f"{name}: noncanonical part order")
        for part in dataset.parts:
            frame = pl.scan_parquet(artifact_path(run.path, part.relative_path))
            if row_count(
                frame.filter(
                    pl.any_horizontal(pl.all().is_null())
                    | (pl.col("run_id") != run.manifest.run_id)
                )
            ):
                raise DataSchemaError(f"{part.relative_path}: null or wrong run_id")
            if index.schema_version == 2 and part.chunk_index >= len(index.chunks):
                raise DataSchemaError(f"{part.relative_path}: has no chunk commitment")
    if run.summary is not None:
        for dataset, field in (
            ("metrics", "metrics_rows"),
            ("births", "birth_rows"),
            ("deaths", "death_rows"),
            ("genealogy", "genealogy_rows"),
        ):
            if field in run.summary and (
                type(run.summary[field]) is not int
                or run.summary[field] != index.datasets[dataset].row_count
            ):
                raise DataSchemaError(f"Summary row count differs from the {dataset} index")
    if index.schema_version == 1:
        return
    parts = {
        name: {p.chunk_index: p for p in dataset.parts} for name, dataset in index.datasets.items()
    }
    end = start
    interval = index.policy.metrics_interval
    for ordinal, chunk in enumerate(index.chunks):
        if (
            chunk.chunk_index != ordinal
            or chunk.start_step != end
            or chunk.start_step >= chunk.end_step_exclusive
            or set(chunk.counts) != set(SCHEMAS)
        ):
            raise DataSchemaError(f"Noncontiguous or malformed committed chunk {ordinal}")
        end = chunk.end_step_exclusive
        for name in SCHEMAS:
            part = parts[name].get(ordinal)
            if chunk.counts[name] != (part.row_count if part else 0) or (
                part and not chunk.start_step <= part.min_step <= part.max_step < end
            ):
                raise DataSchemaError(f"{name}: inconsistent commitment in chunk {ordinal}")
        if scientific:
            first = -(-chunk.start_step // interval) * interval
            if chunk.counts["metrics"] != len(range(first, end, interval)):
                raise DataConsistencyError(f"Missing or extra metric samples in chunk {ordinal}")
    if end > run.manifest.initial_step + run.manifest.total_steps or (
        index.complete and end != run.manifest.initial_step + run.manifest.total_steps
    ):
        raise DataSchemaError("Committed telemetry disagrees with the execution interval")
    expected_end = (
        run.manifest.error.last_committed_telemetry_step
        if run.manifest.error
        else run.manifest.stop.last_committed_telemetry_step
        if run.manifest.stop
        else None
    )
    if expected_end is not None and expected_end != end:
        raise DataSchemaError("Committed telemetry disagrees with the terminal prefix")
