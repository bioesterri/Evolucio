"""Reconstruct demography from the founder census and complete event streams."""

from dataclasses import dataclass
from typing import cast

import polars as pl

from evolucio.persistence.loaders.run_data import RunData

from .types import DEFAULT_ANALYSIS_POLICY, AnalysisPolicy, AnalysisUnavailableError, AnalysisWindow
from .windows import AnalysisRun, count, ratio, validated, window_bounds, window_key, within


def _counts(frame: pl.LazyFrame, step: str, name: str) -> pl.LazyFrame:
    return frame.group_by(pl.col(step).cast(pl.Int64).alias("step")).agg(count().alias(name))


def _trajectory(run: AnalysisRun) -> tuple[pl.LazyFrame, int]:
    if not run.population_available:
        raise AnalysisUnavailableError("founder census or validated trajectory origin unavailable")
    initial = int(run.run.scan_founders().select(count()).collect().item())
    births = _counts(run.run.scan_births(), "birth_step", "birth_count")
    deaths = _counts(run.run.scan_deaths(), "death_step", "death_count")
    changes = births.join(deaths, on="step", how="full", coalesce=True).with_columns(
        pl.col("birth_count", "death_count").fill_null(0)
    )
    return changes.with_columns(
        (pl.col("birth_count") - pl.col("death_count")).alias("net_change")
    ), initial


def population_series(
    run: RunData | AnalysisRun, window: AnalysisWindow | None = None
) -> pl.LazyFrame:
    """Return exact end-of-step populations and fractions per initial agent."""
    data = validated(run)
    coverage = data.bounds(window)
    changes, initial = _trajectory(data)
    anchor = changes.filter(pl.col("step") < coverage.start_step).select(
        (pl.col("net_change").sum() + initial).alias("anchor")
    )
    steps = pl.LazyFrame().select(
        pl.int_range(coverage.start_step, coverage.end_step_exclusive, dtype=pl.Int64).alias("step")
    )
    return (
        steps.join(
            changes.filter(within("step", coverage)), on="step", how="left", maintain_order="left"
        )
        .with_columns(pl.col("birth_count", "death_count", "net_change").fill_null(0))
        .join(anchor, how="cross", maintain_order="left")
        .with_columns((pl.col("anchor") + pl.col("net_change").cum_sum()).alias("population_after"))
        .with_columns(
            pl.lit(data.run.manifest.run_id).alias("run_id"),
            (pl.col("population_after") - pl.col("net_change")).alias("population_before"),
        )
        .drop("anchor")
        .with_columns(
            *(
                ratio(pl.col(field), pl.col("population_before")).alias(name)
                for field, name in (
                    ("birth_count", "birth_fraction_per_step"),
                    ("death_count", "death_fraction_per_step"),
                    ("net_change", "net_growth_fraction_per_step"),
                )
            )
        )
    )


def population_by_window(
    run: RunData | AnalysisRun, window_steps: int, window: AnalysisWindow | None = None
) -> pl.LazyFrame:
    """Aggregate executed steps in absolute, possibly clipped windows."""
    data = validated(run)
    coverage = data.bounds(window)
    after = pl.col("population_after")
    return (
        population_series(data, window)
        .group_by(window_key("step", window_steps))
        .agg(
            count().alias("steps_covered"),
            pl.col("population_before").first().alias("population_start"),
            after.last().alias("population_end"),
            after.min().alias("population_min"),
            after.max().alias("population_max"),
            after.mean().alias("population_mean"),
            after.std(ddof=0).alias("population_std"),
            pl.col("birth_count").sum().alias("births_total"),
            pl.col("death_count").sum().alias("deaths_total"),
            pl.col("net_change").sum(),
            pl.col("population_before").sum().alias("agent_steps_at_start"),
        )
        .with_columns(*window_bounds(window_steps, coverage))
        .with_columns(
            (pl.col("steps_covered") < window_steps).alias("window_incomplete"),
            (pl.col("population_max") - pl.col("population_min")).alias("population_range"),
            ratio(pl.col("population_std"), pl.col("population_mean")).alias("population_cv"),
            *(
                ratio(pl.col(f"{event}_total"), pl.col("agent_steps_at_start")).alias(
                    f"{event}_per_agent_step"
                )
                for event in ("births", "deaths")
            ),
        )
        .sort("window_start")
    )


@dataclass(frozen=True, slots=True)
class PopulationSummary:
    initial_population: int
    final_population: int
    minimum_population: int | None
    maximum_population: int | None
    mean_population: float | None
    median_population: float | None
    population_std: float | None
    population_cv: float | None
    population_range: int | None
    births_total: int
    deaths_total: int
    net_population_change: int
    agent_steps_at_start: int
    population_extinct: bool
    extinction_state_step: int | None
    time_to_extinction: int | None
    population_capacity: int
    coverage_start: int
    coverage_end_exclusive: int


def population_summary(
    run: RunData | AnalysisRun, window: AnalysisWindow | None = None
) -> PopulationSummary:
    data = validated(run)
    coverage = data.bounds(window)
    changes, initial = _trajectory(data)
    origin = data.coverage.start_step
    extinction = (
        origin
        if initial == 0
        else cast(
            int | None,
            changes.filter(pl.col("step") < coverage.end_step_exclusive)
            .sort("step")
            .filter(pl.col("net_change").cum_sum() + initial == 0)
            .select(pl.col("step").min() + 1)
            .collect(engine="streaming")
            .item(),
        )
    )
    after = pl.col("population_after")
    stats = (
        population_series(data, window)
        .select(
            pl.col("population_before").first().fill_null(initial).alias("initial_population"),
            after.last().fill_null(initial).alias("final_population"),
            after.min().alias("minimum_population"),
            after.max().alias("maximum_population"),
            after.mean().alias("mean_population"),
            after.median().alias("median_population"),
            after.std(ddof=0).alias("population_std"),
            ratio(after.std(ddof=0), after.mean()).alias("population_cv"),
            (after.max() - after.min()).alias("population_range"),
            pl.col("birth_count").sum().alias("births_total"),
            pl.col("death_count").sum().alias("deaths_total"),
            pl.col("net_change").sum().alias("net_population_change"),
            pl.col("population_before").sum().alias("agent_steps_at_start"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    return PopulationSummary(
        **stats,
        population_extinct=extinction is not None,
        extinction_state_step=extinction,
        time_to_extinction=None if extinction is None else extinction - origin,
        population_capacity=data.run.config.population.max_agents,
        coverage_start=coverage.start_step,
        coverage_end_exclusive=coverage.end_step_exclusive,
    )


def bottlenecks(
    run: RunData | AnalysisRun,
    policy: AnalysisPolicy = DEFAULT_ANALYSIS_POLICY,
    window: AnalysisWindow | None = None,
) -> pl.LazyFrame:
    data = validated(run)
    coverage = data.bounds(window)
    _, initial = _trajectory(data)
    threshold = policy.bottleneck_fraction * initial
    after = pl.col("population_after")
    return (
        population_series(data, window)
        .with_columns(((after > 0) & (after < threshold)).alias("below"))
        .with_columns(pl.col("below").rle_id().alias("episode"))
        .filter(pl.col("below"))
        .group_by("episode")
        .agg(
            (pl.col("step").min() + 1).alias("episode_start_state_step"),
            (pl.col("step").max() + 2).alias("episode_end_state_step_exclusive"),
            count().alias("duration_observed"),
            after.min().alias("minimum_population"),
        )
        .drop("episode")
        .with_columns(
            pl.lit(threshold).alias("threshold"),
            (pl.col("episode_end_state_step_exclusive") == coverage.end_step_exclusive + 1).alias(
                "right_censored"
            ),
        )
        .sort("episode_start_state_step")
    )
