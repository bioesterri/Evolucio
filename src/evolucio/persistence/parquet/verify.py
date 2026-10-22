"""Read-only byte, schema and biological projection verification."""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from evolucio.persistence.local.artifacts import artifact_path, sha256_file

from .index import load_index
from .reader import scan_dataset
from .schemas import GENEALOGY_COLUMNS, SCHEMAS, STEP_COLUMNS


@dataclass(frozen=True, slots=True)
class ParquetDatasetVerification:
    ok: bool
    missing_parts: tuple[str, ...]
    hash_mismatches: tuple[str, ...]
    schema_mismatches: tuple[str, ...]
    orphan_parts: tuple[str, ...]
    consistency_errors: tuple[str, ...]


def consistency_errors(run_dir: Path) -> list[str]:
    index = load_index(run_dir)
    errors: list[str] = []
    for name, identity in (
        ("metrics", "step"),
        ("births", "child_agent_id"),
        ("deaths", "agent_id"),
        ("genealogy", "child_agent_id"),
    ):
        frame = scan_dataset(run_dir, name)
        counts = frame.select(
            pl.len().alias("rows"),
            pl.col(identity).n_unique().alias("ids"),
            (pl.col("run_id") != index.run_id).any().alias("wrong_run"),
        ).collect(engine="streaming")
        if counts["rows"][0] != counts["ids"][0] or counts["wrong_run"][0]:
            errors.append(f"{name}: duplicate identity or wrong run_id")
    metrics = scan_dataset(run_dir, "metrics")
    if (
        metrics.filter(pl.col("step") % index.policy.metrics_interval != 0)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    ):
        errors.append("metrics: invalid sampling")
    births = scan_dataset(run_dir, "births").select(GENEALOGY_COLUMNS)
    genealogy = scan_dataset(run_dir, "genealogy")
    for left, right in ((births, genealogy), (genealogy, births)):
        if (
            left.join(right, on=list(GENEALOGY_COLUMNS), how="anti")
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        ):
            errors.append("births/genealogy projection mismatch")
            break
    deaths = scan_dataset(run_dir, "deaths")
    if (
        deaths.join(births, left_on="agent_id", right_on="child_agent_id")
        .filter(pl.col("death_step") < pl.col("birth_step"))
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    ):
        errors.append("death precedes birth")
    return errors


def verify_parquet_datasets(run_dir: Path) -> ParquetDatasetVerification:
    run_dir = run_dir.resolve()
    missing: list[str] = []
    hashes: list[str] = []
    schemas: list[str] = []
    errors: list[str] = []
    indexed: set[str] = set()
    try:
        index = load_index(run_dir)
        for name, dataset in index.datasets.items():
            rows = 0
            minima: list[int] = []
            maxima: list[int] = []
            for part in dataset.parts:
                relative = part.relative_path
                indexed.add(relative)
                path = artifact_path(run_dir, relative)
                if not path.is_file():
                    missing.append(relative)
                    continue
                if path.stat().st_size != part.byte_size or sha256_file(path) != part.sha256:
                    hashes.append(relative)
                    continue
                try:
                    frame = pl.scan_parquet(path)
                    if frame.collect_schema() != SCHEMAS[name]:
                        schemas.append(relative)
                        continue
                    stats = (
                        frame.select(
                            pl.len().alias("rows"),
                            pl.col(STEP_COLUMNS[name]).min().alias("min"),
                            pl.col(STEP_COLUMNS[name]).max().alias("max"),
                        )
                        .collect(engine="streaming")
                        .row(0)
                    )
                    if stats != (part.row_count, part.min_step, part.max_step):
                        errors.append(f"{relative}: part statistics mismatch")
                    rows += part.row_count
                    minima.append(part.min_step)
                    maxima.append(part.max_step)
                except (OSError, pl.exceptions.PolarsError) as exc:
                    schemas.append(f"{relative}: {exc}")
            if (
                dataset.row_count != rows
                or dataset.part_count != len(dataset.parts)
                or dataset.min_step != (min(minima) if minima else None)
                or dataset.max_step != (max(maxima) if maxima else None)
            ):
                errors.append(f"{name}: dataset statistics mismatch")
        if not (missing or hashes or schemas):
            errors.extend(consistency_errors(run_dir))
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        errors.append(str(exc))
    orphans: list[str] = []
    try:
        root = artifact_path(run_dir, "artifacts/parquet")
        orphans = sorted(
            str(path.relative_to(run_dir))
            for path in root.rglob("*.parquet")
            if str(path.relative_to(run_dir)) not in indexed
        )
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    return ParquetDatasetVerification(
        not (missing or hashes or schemas or orphans or errors),
        tuple(missing),
        tuple(hashes),
        tuple(schemas),
        tuple(orphans),
        tuple(errors),
    )
