"""Shared validation and absolute-step window expressions."""

from dataclasses import dataclass

import polars as pl

from evolucio.analysis.summaries import FOUNDERS_PATH
from evolucio.persistence.loaders.run_data import RunData
from evolucio.persistence.local.manifest import RunStatus
from evolucio.persistence.validation.runner import COMPONENTS, evaluate_run
from evolucio.persistence.validation.types import CheckStatus, ValidationLevel, ValidationReport

from .types import AnalysisUnavailableError, AnalysisWindow, Coverage, positive_int


@dataclass(frozen=True, slots=True)
class AnalysisRun:
    run: RunData
    report: ValidationReport
    coverage: Coverage
    metrics_interval: int

    def bounds(self, window: AnalysisWindow | None) -> Coverage:
        if window is None:
            return self.coverage
        if (
            window.start_step < self.coverage.start_step
            or window.end_step_exclusive > self.coverage.end_step_exclusive
        ):
            raise ValueError("analysis window is outside committed telemetry")
        return Coverage(window.start_step, window.end_step_exclusive)

    @property
    def population_available(self) -> bool:
        return self.report.components["population"] is CheckStatus.PASSED and self.has_artifact(
            FOUNDERS_PATH
        )

    def has_artifact(self, path: str) -> bool:
        return any(record.relative_path == path for record in self.run.manifest.artifacts)


def validated(run: RunData | AnalysisRun) -> AnalysisRun:
    if isinstance(run, AnalysisRun):
        return run
    data, report, error = evaluate_run(
        run.path,
        validation_level=ValidationLevel.SCIENTIFIC,
        components=tuple(name for name in COMPONENTS if name != "genomes"),
        allow_partial=True,
    )
    if data is None or not report.local_validation_passed:
        detail = "; ".join(issue.message for issue in report.issues)
        raise ValueError(f"Scientific validation failed: {detail}") from error
    if data.manifest.status not in (RunStatus.COMPLETED, RunStatus.STOPPED, RunStatus.FAILED):
        raise AnalysisUnavailableError("analysis requires a terminal run")
    index = data.parquet
    if index is None or report.components["telemetry_coverage"] is not CheckStatus.PASSED:
        raise AnalysisUnavailableError("confirmed Parquet coverage is unavailable")
    start = 0 if data.recovery else data.manifest.initial_step
    end = index.chunks[-1].end_step_exclusive if index.chunks else start
    return AnalysisRun(data, report, Coverage(start, end), index.policy.metrics_interval)


def within(column: str, coverage: Coverage) -> pl.Expr:
    return pl.col(column).is_between(
        coverage.start_step, coverage.end_step_exclusive, closed="left"
    )


def window_key(column: str, width: int) -> pl.Expr:
    positive_int(width, "window_steps")
    return (pl.col(column).cast(pl.Int64) // width * width).alias("window_start")


def window_bounds(width: int, coverage: Coverage) -> tuple[pl.Expr, ...]:
    return (
        pl.max_horizontal("window_start", pl.lit(coverage.start_step)).alias("covered_start"),
        pl.min_horizontal(
            pl.col("window_start") + width, pl.lit(coverage.end_step_exclusive)
        ).alias("window_end_exclusive"),
    )


def ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return numerator / pl.when(denominator != 0).then(denominator)


def count() -> pl.Expr:
    return pl.len().cast(pl.Int64)


def window_grid(width: int, coverage: Coverage) -> pl.LazyFrame:
    positive_int(width, "window_steps")
    start = coverage.start_step
    end = coverage.end_step_exclusive
    first = start // width * width if start < end else end
    return (
        pl.LazyFrame()
        .select(pl.int_range(first, end, step=width, dtype=pl.Int64).alias("window_start"))
        .with_columns(*window_bounds(width, coverage))
        .with_columns(
            (pl.col("window_end_exclusive") - pl.col("covered_start")).alias("steps_covered")
        )
        .with_columns((pl.col("steps_covered") < width).alias("window_incomplete"))
    )
