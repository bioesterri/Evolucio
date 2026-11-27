"""Absolute-step alignment with one observation per trajectory and step."""

from collections.abc import Mapping
from enum import StrEnum

import polars as pl

from evolucio.analysis.energy import energy_series
from evolucio.analysis.genetic_diversity import genetic_diversity_series
from evolucio.analysis.population import population_series
from evolucio.analysis.types import AnalysisWindow
from evolucio.analysis.windows import AnalysisRun
from evolucio.experiments.design import CaseRole, ResolvedDesign

from .selection import Cell
from .statistics import descriptive


class AlignmentPolicy(StrEnum):
    EXACT = "EXACT"
    COMMON_STEPS = "COMMON_STEPS"


def aligned_series(
    cases: Mapping[Cell, AnalysisRun],
    metric_name: str,
    window: AnalysisWindow,
    alignment_policy: AlignmentPolicy = AlignmentPolicy.EXACT,
) -> pl.LazyFrame:
    if not cases:
        raise ValueError("at least one verified run is required")
    frames: list[pl.LazyFrame] = []
    for (case_id, seed), run in cases.items():
        run.bounds(window)
        if metric_name in {"population_before", "population_after", "birth_count", "death_count"}:
            frame = population_series(run, window)
        elif metric_name in {"genetic_parameter_variance", "pairwise_rms_distance_per_parameter"}:
            frame = genetic_diversity_series(run, window).with_columns(
                pl.when(pl.col("population") > 0).then(pl.col(metric_name)).alias(metric_name)
            )
        else:
            frame = energy_series(run, window)
            if metric_name in {"step", "run_id"} or metric_name not in frame.collect_schema():
                raise ValueError(f"unsupported temporal metric: {metric_name}")
        frames.append(
            frame.select(
                "step",
                "run_id",
                pl.col(metric_name).cast(pl.Float64).alias("value"),
                pl.lit(case_id).alias("case_id"),
                pl.lit(seed, dtype=pl.Int64).alias("seed"),
            )
        )
    combined = pl.concat(frames)
    counts = (
        combined.group_by("step")
        .agg(pl.len().alias("rows"), pl.col("run_id").n_unique().alias("runs"))
        .collect()
    )
    if counts.filter(pl.col("rows") != pl.col("runs")).height:
        raise ValueError("duplicate trajectory observation at a step")
    common = counts.filter(pl.col("runs") == len(cases)).select("step")
    if alignment_policy is AlignmentPolicy.EXACT and common.height != counts.height:
        raise ValueError("observation steps differ")
    if not common.height:
        raise ValueError("no common observed steps")
    return combined.join(common.lazy(), on="step", how="semi")


def compare_time_series(
    cases: Mapping[Cell, AnalysisRun],
    metric_name: str,
    window: AnalysisWindow,
    alignment_policy: AlignmentPolicy = AlignmentPolicy.EXACT,
) -> pl.LazyFrame:
    return summarize_time_series(aligned_series(cases, metric_name, window, alignment_policy))


def summarize_time_series(series: pl.LazyFrame) -> pl.LazyFrame:
    return (
        series.group_by("case_id", "step")
        .agg(pl.col("value").count().alias("n_available_runs"), *descriptive("value"))
        .sort("case_id", "step")
    )


def paired_time_series(series: pl.LazyFrame, baseline_case_id: str) -> pl.LazyFrame:
    baseline = series.filter(pl.col("case_id") == baseline_case_id).select(
        "seed", "step", pl.col("value").alias("baseline")
    )
    return (
        series.filter(pl.col("case_id") != baseline_case_id)
        .join(baseline, on=["seed", "step"], validate="m:1")
        .with_columns((pl.col("value") - pl.col("baseline")).alias("delta"))
        .group_by("case_id", "step")
        .agg(
            pl.col("delta").count().alias("n_pairs_available"),
            *descriptive("delta", suffix="_difference"),
        )
        .sort("case_id", "step")
    )


def differential_change(
    before: pl.LazyFrame,
    after: pl.LazyFrame,
    baseline_case_id: str,
    control_case_id: str,
) -> pl.LazyFrame:
    """Contrast baseline minus control changes for the same seed."""
    keys = ["case_id", "seed"]
    changes = (
        before.group_by(*keys)
        .agg(
            pl.when(pl.col("value").null_count() == 0).then(pl.col("value").mean()).alias("before")
        )
        .join(
            after.group_by(*keys).agg(
                pl.when(pl.col("value").null_count() == 0)
                .then(pl.col("value").mean())
                .alias("after")
            ),
            on=keys,
            how="full",
            coalesce=True,
            validate="1:1",
        )
        .with_columns((pl.col("after") - pl.col("before")).alias("change"))
    )

    def side(case_id: str, label: str) -> pl.LazyFrame:
        return changes.filter(pl.col("case_id") == case_id).select(
            "seed", pl.col("change").alias(label)
        )

    return (
        side(baseline_case_id, "change_baseline")
        .join(
            side(control_case_id, "change_control"),
            on="seed",
            how="full",
            coalesce=True,
            validate="1:1",
        )
        .with_columns(
            (pl.col("change_baseline") - pl.col("change_control")).alias("differential_change")
        )
        .sort("seed")
    )


def environment_windows(
    resolved: ResolvedDesign,
    control_case_id: str,
) -> dict[str, AnalysisWindow]:
    """Resolve windows from the actual baseline phase and a constant control."""
    design = resolved.design
    reference = design.environment_change_step
    if reference is None:
        raise ValueError("the design needs an environment_change_step")
    configs = {
        spec.case_id: config.config.world
        for spec, config in zip(resolved.batch.specs, resolved.batch.configs, strict=True)
        if spec.seed == design.seed_set[0]
    }
    baseline = configs[design.baseline_case_id]
    control = configs.get(control_case_id)
    if (
        control is None
        or not any(
            case.case_id == control_case_id and case.role is CaseRole.CONTROL
            for case in design.cases
        )
        or control.environment_schedule
        or control.environment_initial_value != baseline.environment_initial_value
    ):
        raise ValueError("expected a constant control with the baseline initial environment")
    phase = next(item for item in baseline.environment_schedule if item.start_step == reference)
    if phase.end_step >= design.total_steps:
        raise ValueError("no post-perturbation window within the planned horizon")
    return {
        "before": AnalysisWindow(0, reference),
        "during": AnalysisWindow(reference, phase.end_step),
        "after": AnalysisWindow(phase.end_step, design.total_steps),
    }
