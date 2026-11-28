"""Descriptive statistics over independent runs and matched seed blocks."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import cast

import numpy as np
import polars as pl

from evolucio.analysis.types import positive_int

from .metrics import METRICS

METRIC_KEYS = ["metric_name", "dimension"]


def descriptive(column: str, *, suffix: str = "") -> tuple[pl.Expr, ...]:
    value = pl.col(column)
    return tuple(
        expr.alias(name + suffix)
        for name, expr in (
            ("mean", value.mean()),
            ("median", value.median()),
            ("sample_std", value.std(ddof=1)),
            ("minimum", value.min()),
            ("maximum", value.max()),
            ("q25", value.quantile(0.25, interpolation="linear")),
            ("q75", value.quantile(0.75, interpolation="linear")),
        )
    )


def _long_metrics(frame: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    result = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    names = result.collect_schema().names()
    if "dimension" not in names:
        result = result.with_columns(pl.lit("").alias("dimension"))
    if "availability_status" not in names:
        result = result.with_columns(
            pl.when(pl.col("metric_value").is_not_null())
            .then(pl.lit("defined"))
            .otherwise(pl.lit("unavailable"))
            .alias("availability_status")
        )
    return result


def summarize_case_metrics(run_metrics: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    return (
        _long_metrics(run_metrics)
        .group_by("case_id", *METRIC_KEYS)
        .agg(
            pl.len().alias("n_expected"),
            pl.col("metric_value").count().alias("n_available"),
            *descriptive("metric_value"),
        )
        .with_columns(
            (pl.col("n_expected") - pl.col("n_available")).alias("n_missing"),
            (pl.col("q75") - pl.col("q25")).alias("iqr"),
        )
        .sort("case_id", *METRIC_KEYS)
    )


def compare_paired_metrics(
    baseline: pl.DataFrame | pl.LazyFrame,
    variant: pl.DataFrame | pl.LazyFrame,
    metric_name: str,
) -> pl.LazyFrame:
    definition = METRICS[metric_name]
    keys = ["seed", *METRIC_KEYS]

    def side(frame: pl.DataFrame | pl.LazyFrame, prefix: str) -> pl.LazyFrame:
        return (
            _long_metrics(frame)
            .filter(pl.col("metric_name") == metric_name)
            .select(
                *keys,
                pl.col("run_id").alias(f"{prefix}_run_id"),
                pl.col("metric_value").cast(pl.Float64).alias(f"{prefix}_value"),
                pl.col("availability_status").alias(f"{prefix}_availability"),
            )
        )

    delta = pl.col("variant_value") - pl.col("baseline_value")
    return (
        side(baseline, "baseline")
        .join(side(variant, "variant"), on=keys, how="full", coalesce=True, validate="1:1")
        .with_columns(
            delta.alias("absolute_difference"),
            pl.when(pl.lit(definition.ratio_scale) & (pl.col("baseline_value") > 0))
            .then(delta / pl.col("baseline_value"))
            .alias("relative_difference"),
        )
        .sort(*keys)
    )


def summarize_paired(differences: pl.LazyFrame, tolerance: float) -> pl.LazyFrame:
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    delta = pl.col("absolute_difference")
    signs = {
        "positive": delta > tolerance,
        "negative": delta < -tolerance,
        "equal": delta.abs() <= tolerance,
    }
    return (
        differences.group_by("variant_case_id", *METRIC_KEYS)
        .agg(
            delta.count().alias("n_pairs"),
            pl.len().alias("n_expected"),
            *descriptive("absolute_difference", suffix="_difference"),
            *(expr.sum().alias(f"{sign}_difference_count") for sign, expr in signs.items()),
        )
        .with_columns(
            *(
                (
                    pl.col(f"{sign}_difference_count")
                    / pl.when(pl.col("n_pairs") > 0).then(pl.col("n_pairs"))
                ).alias(f"{sign}_difference_fraction")
                for sign in signs
            )
        )
        .sort("variant_case_id", *METRIC_KEYS)
    )


@dataclass(frozen=True, slots=True)
class BootstrapPolicy:
    seed: int
    resamples: int = 10_000
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        positive_int(self.resamples, "resamples")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("bootstrap seed must be a nonnegative integer")
        if type(self.confidence_level) is bool or not 0 < self.confidence_level < 1:
            raise ValueError("confidence level must be between zero and one")


def bootstrap_paired_mean(
    deltas: Sequence[float],
    policy: BootstrapPolicy,
) -> tuple[float, float] | None:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("expected finite one-dimensional paired deltas")
    if len(values) < 2:
        return None
    rng = np.random.default_rng(policy.seed)
    means = np.empty(policy.resamples)
    batch_size = max(1, 65_536 // len(values))
    for start in range(0, policy.resamples, batch_size):
        stop = min(policy.resamples, start + batch_size)
        indices = rng.integers(len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    tail = (1 - policy.confidence_level) / 2
    low, high = np.quantile(means, (tail, 1 - tail), method="linear")
    return float(low), float(high)


def bootstrap_summaries(
    differences: pl.LazyFrame,
    policy: BootstrapPolicy,
) -> pl.LazyFrame:
    keys = ["variant_case_id", *METRIC_KEYS]
    groups = (
        differences.sort("seed")
        .group_by(*keys)
        .agg(pl.col("absolute_difference").drop_nulls().alias("deltas"))
        .collect()
    )
    rows: list[dict[str, object]] = []
    for row in groups.iter_rows(named=True):
        interval = bootstrap_paired_mean(cast(list[float], row["deltas"]), policy)
        rows.append(
            {
                **{key: row[key] for key in keys},
                "bootstrap_low": None if interval is None else interval[0],
                "bootstrap_high": None if interval is None else interval[1],
            }
        )
    return pl.DataFrame(
        rows,
        schema=dict.fromkeys(keys, pl.String)
        | {"bootstrap_low": pl.Float64, "bootstrap_high": pl.Float64},
    ).lazy()


def extinction_tables(
    metrics: pl.LazyFrame,
    differences: pl.LazyFrame,
    total_steps: int,
) -> dict[str, pl.LazyFrame]:
    extinct = pl.col("metric_value")
    cases = (
        metrics.filter(pl.col("metric_name") == "population_extinct")
        .group_by("case_id")
        .agg(
            pl.len().alias("n_expected"),
            extinct.count().alias("n_completed"),
            (extinct == 1).sum().alias("n_extinct"),
            (extinct == 0).sum().alias("n_not_extinct"),
        )
        .with_columns(
            (
                pl.col("n_extinct") / pl.when(pl.col("n_completed") > 0).then(pl.col("n_completed"))
            ).alias("observed_extinction_fraction"),
        )
        .sort("case_id")
    )
    baseline, variant = pl.col("baseline_value"), pl.col("variant_value")
    pairs = differences.filter(
        (pl.col("metric_name") == "population_extinct")
        & pl.col("pair_eligible")
        & baseline.is_not_null()
        & variant.is_not_null()
    ).with_columns(
        pl.when((baseline == 1) & (variant == 1))
        .then(pl.lit("both_extinct"))
        .when(baseline == 1)
        .then(pl.lit("baseline_only_extinct"))
        .when(variant == 1)
        .then(pl.lit("variant_only_extinct"))
        .otherwise(pl.lit("neither_extinct"))
        .alias("category")
    )
    times = (
        metrics.filter((pl.col("metric_name") == "population_extinct") & extinct.is_not_null())
        .select("case_id", "seed", "run_id", (extinct == 1).alias("event_observed"))
        .join(
            metrics.filter(pl.col("metric_name") == "time_to_extinction").select(
                "case_id", "seed", pl.col("metric_value").alias("event_time")
            ),
            on=["case_id", "seed"],
            validate="1:1",
        )
        .with_columns(
            pl.when(pl.col("event_observed"))
            .then(pl.col("event_time"))
            .otherwise(pl.lit(total_steps))
            .cast(pl.Int64)
            .alias("observed_age")
        )
        .sort("case_id", "seed")
    )
    categories = pl.LazyFrame(
        {
            "category": [
                "both_extinct",
                "baseline_only_extinct",
                "variant_only_extinct",
                "neither_extinct",
            ]
        }
    )
    paired_counts = (
        differences.select("variant_case_id")
        .unique()
        .join(categories, how="cross")
        .join(
            pairs.group_by("variant_case_id", "category").len().rename({"len": "n_pairs"}),
            on=["variant_case_id", "category"],
            how="left",
            validate="1:1",
        )
        .with_columns(pl.col("n_pairs").fill_null(0))
        .sort("variant_case_id", "category")
    )
    return {
        "extinction_summary": cases,
        "paired_extinction": pairs,
        "paired_extinction_counts": paired_counts,
        "extinction_observations": times,
    }
