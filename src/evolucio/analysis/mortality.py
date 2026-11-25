"""Mortality from complete events, including diagnostic deaths."""

from collections.abc import Sequence
from itertools import pairwise
from math import isfinite

import polars as pl

from evolucio.core.codes import DeathCauseCode
from evolucio.core.metrics.events import DeathPhaseCode
from evolucio.persistence.loaders.run_data import RunData

from .population import population_by_window
from .types import AnalysisWindow
from .windows import AnalysisRun, count, ratio, validated, window_grid, window_key, within


def _causes() -> pl.LazyFrame:
    causes = [cause for cause in DeathCauseCode if cause is not DeathCauseCode.NONE]
    return pl.LazyFrame(
        {
            "cause_code": [int(cause) for cause in causes],
            "cause_name": [cause.name for cause in causes],
        }
    )


def _deaths(run: AnalysisRun, window: AnalysisWindow | None) -> pl.LazyFrame:
    return (
        run.run.scan_deaths()
        .filter(within("death_step", run.bounds(window)))
        .with_columns(pl.col("cause_code", "phase_code").cast(pl.Int64))
    )


def mortality_by_cause(
    run: RunData | AnalysisRun, window: AnalysisWindow | None = None
) -> pl.LazyFrame:
    deaths = (
        _deaths(validated(run), window).group_by("cause_code").agg(count().alias("death_count"))
    )
    return (
        _causes()
        .join(deaths, on="cause_code", how="left")
        .with_columns(pl.col("death_count").fill_null(0))
        .with_columns(
            ratio(pl.col("death_count"), pl.col("death_count").sum()).alias("death_fraction")
        )
        .sort("cause_code")
    )


def mortality_by_window(
    run: RunData | AnalysisRun, window_steps: int, window: AnalysisWindow | None = None
) -> pl.LazyFrame:
    data = validated(run)
    windows = window_grid(window_steps, data.bounds(window))
    if data.population_available:
        exposure = population_by_window(data, window_steps, window).select(
            "window_start", "agent_steps_at_start"
        )
        windows = windows.join(exposure, on="window_start", how="left")
    else:
        windows = windows.with_columns(pl.lit(None, dtype=pl.Int64).alias("agent_steps_at_start"))
    observed = (
        _deaths(data, window)
        .group_by(window_key("death_step", window_steps), "cause_code")
        .agg(count().alias("death_count"))
    )
    return (
        windows.join(_causes(), how="cross")
        .join(observed, on=["window_start", "cause_code"], how="left")
        .with_columns(pl.col("death_count").fill_null(0))
        .with_columns(pl.col("death_count").sum().over("window_start").alias("total_deaths"))
        .with_columns(
            ratio(pl.col("death_count"), pl.col("agent_steps_at_start")).alias(
                "deaths_per_agent_step"
            )
        )
        .sort("window_start", "cause_code")
    )


def mortality_by_phase(
    run: RunData | AnalysisRun, window: AnalysisWindow | None = None
) -> pl.LazyFrame:
    frames = _deaths(validated(run), window)
    phases = {
        int(phase): phase.name for phase in DeathPhaseCode if phase is not DeathPhaseCode.NONE
    }
    return (
        frames.group_by("cause_code", "phase_code")
        .agg(count().alias("death_count"))
        .join(_causes(), on="cause_code")
        .with_columns(pl.col("phase_code").replace_strict(phases).alias("phase_name"))
        .sort("cause_code", "phase_code")
    )


def _distribution(value: pl.Expr) -> tuple[pl.Expr, ...]:
    return (
        value.mean().alias("mean"),
        value.median().alias("median"),
        value.min().alias("minimum"),
        value.max().alias("maximum"),
        value.std(ddof=0).alias("std"),
        value.quantile(0.25, interpolation="linear").alias("q25"),
        value.quantile(0.75, interpolation="linear").alias("q75"),
    )


def death_age_distribution(
    run: RunData | AnalysisRun,
    window: AnalysisWindow | None = None,
    *,
    by_cause: bool = False,
) -> pl.LazyFrame:
    """Describe observed deaths with linearly interpolated quartiles."""
    deaths = _deaths(validated(run), window)
    stats = (count().alias("count"), *_distribution(pl.col("age").cast(pl.Int64)))
    if by_cause:
        return (
            deaths.group_by("cause_code")
            .agg(*stats)
            .join(_causes(), on="cause_code")
            .sort("cause_code")
        )
    return deaths.select(*stats)


def death_energy_distribution(
    run: RunData | AnalysisRun, window: AnalysisWindow | None = None
) -> pl.LazyFrame:
    energy = pl.col("energy").cast(pl.Float64)
    finite = energy.is_finite()
    return (
        _deaths(validated(run), window)
        .group_by("cause_code", "phase_code")
        .agg(
            finite.sum().cast(pl.Int64).alias("valid_energy_count"),
            (~finite).sum().cast(pl.Int64).alias("invalid_energy_count"),
            *_distribution(energy.filter(finite)),
        )
        .sort("cause_code", "phase_code")
    )


def _histogram(deaths: pl.LazyFrame, column: str, edges: Sequence[float]) -> pl.LazyFrame:
    if (
        not edges
        or any(not isfinite(edge) for edge in edges)
        or any(lower >= upper for lower, upper in pairwise(edges))
    ):
        raise ValueError("histogram edges must be finite and strictly increasing")
    value = pl.col(column).cast(pl.Float64)
    bins = [(None, edges[0]), *pairwise(edges), (edges[-1], None)]
    conditions = [
        (
            (value >= lower if lower is not None else pl.lit(True))
            & (value < upper if upper is not None else pl.lit(True))
            & value.is_finite()
        )
        for lower, upper in bins
    ]
    names = [str(index) for index in range(len(bins))]
    counts = deaths.select(
        *(
            condition.sum().cast(pl.Int64).alias(name)
            for name, condition in zip(names, conditions, strict=True)
        ),
        (~value.is_finite()).sum().cast(pl.Int64).alias("invalid_count"),
    ).unpivot(on=names, index="invalid_count", variable_name="bin", value_name="count")
    return (
        pl.LazyFrame(
            {
                "bin": names,
                "lower_inclusive": [low for low, _ in bins],
                "upper_exclusive": [high for _, high in bins],
            },
            schema_overrides={"lower_inclusive": pl.Float64, "upper_exclusive": pl.Float64},
        )
        .join(counts, on="bin", maintain_order="left")
        .drop("bin")
    )


def death_age_histogram(
    run: RunData | AnalysisRun, edges: Sequence[float], window: AnalysisWindow | None = None
) -> pl.LazyFrame:
    return _histogram(_deaths(validated(run), window), "age", edges)


def death_energy_histogram(
    run: RunData | AnalysisRun, edges: Sequence[float], window: AnalysisWindow | None = None
) -> pl.LazyFrame:
    return _histogram(_deaths(validated(run), window), "energy", edges)
