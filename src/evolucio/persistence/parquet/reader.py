"""Lazy analytical readers use the validated index, never filesystem globs."""

from pathlib import Path

import polars as pl

from evolucio.persistence.local.artifacts import artifact_path

from .index import load_index
from .schemas import SCHEMAS


def scan_dataset(run_dir: Path, name: str) -> pl.LazyFrame:
    index = load_index(run_dir)
    parts = index.datasets[name].parts
    if not parts:
        return pl.DataFrame(schema=SCHEMAS[name]).lazy()
    return pl.scan_parquet([artifact_path(run_dir, part.relative_path) for part in parts])


def scan_metrics(run_dir: Path) -> pl.LazyFrame:
    return scan_dataset(run_dir, "metrics")


def scan_births(run_dir: Path) -> pl.LazyFrame:
    return scan_dataset(run_dir, "births")


def scan_deaths(run_dir: Path) -> pl.LazyFrame:
    return scan_dataset(run_dir, "deaths")


def scan_genealogy(run_dir: Path) -> pl.LazyFrame:
    return scan_dataset(run_dir, "genealogy")
