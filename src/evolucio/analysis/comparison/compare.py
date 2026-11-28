"""End-to-end experimental comparison over persisted trajectories."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite
from uuid import uuid4

import polars as pl

from evolucio.experiments.design import ExperimentDesign, PairPolicy, ResolvedDesign, resolve_design

from .compatibility import ComparabilityReport, compare_compatibility
from .metrics import Observation, extract_metrics
from .selection import Cell, LocalRunProvider, RunProvider, SelectionReport, select_comparison_runs
from .statistics import (
    METRIC_KEYS,
    BootstrapPolicy,
    bootstrap_summaries,
    compare_paired_metrics,
    extinction_tables,
    summarize_case_metrics,
    summarize_paired,
)

COMPARISON_SCHEMA_VERSION = 1
DEFAULT_TOLERANCE = 1e-6
RUN_METRICS_SCHEMA = {
    **dict.fromkeys(
        (
            "experiment_id",
            "case_id",
            "role",
            "run_id",
            "root_run_id",
            "config_hash",
            "metric_name",
            "dimension",
            "metric_quality",
            "availability_status",
            "availability_reason",
        ),
        pl.String,
    ),
    "seed": pl.Int64,
    "metric_value": pl.Float64,
    "coverage_start": pl.Int64,
    "coverage_end_exclusive": pl.Int64,
}


@dataclass(frozen=True, slots=True)
class ExperimentComparisonResult:
    comparison_id: str
    schema_version: int
    experiment_design: ResolvedDesign
    comparison_policy: PairPolicy
    interpretation: str
    run_selection_report: SelectionReport
    comparability_reports: tuple[ComparabilityReport, ...]
    tables: dict[str, pl.LazyFrame]
    warnings: tuple[str, ...]
    unavailable_metrics: dict[str, int]
    tolerance: float
    bootstrap: BootstrapPolicy | None

    @property
    def experiment_id(self) -> str:
        return str(self.experiment_design.design.experiment_id)

    @property
    def baseline_case_id(self) -> str:
        return self.experiment_design.design.baseline_case_id

    @property
    def compared_case_ids(self) -> tuple[str, ...]:
        return tuple(report.variant_case_id for report in self.comparability_reports)

    @property
    def case_summaries(self) -> pl.LazyFrame:
        return self.tables["case_summaries"]

    @property
    def paired_difference_summaries(self) -> pl.LazyFrame:
        return self.tables["paired_summaries"]

    @property
    def extinction_summaries(self) -> pl.LazyFrame:
        return self.tables["extinction_summary"]


def _coverage_report(selection: SelectionReport) -> pl.LazyFrame:
    return (
        pl.DataFrame(
            [
                {
                    "case_id": case,
                    "seed": seed,
                    "status": state,
                    "completed": (case, seed) in selection.completed_cells,
                }
                for (case, seed), state in selection.cells.items()
            ]
        )
        .lazy()
        .group_by("case_id")
        .agg(
            pl.len().alias("expected_seed_count"),
            pl.col("completed").sum().alias("completed_seed_count"),
            *(
                (pl.col("status") == state).sum().alias(f"{label}_seed_count")
                for state, label in (
                    ("completed", "eligible"),
                    ("failed", "failed"),
                    ("stopped", "stopped"),
                    ("unknown", "missing"),
                )
            ),
            (pl.col("status") != "completed").sum().alias("excluded_seed_count"),
        )
        .sort("case_id")
    )


def _run_metrics(
    design: ExperimentDesign,
    selection: SelectionReport,
    names: tuple[str, ...],
) -> pl.LazyFrame:
    from evolucio.core.codes import DeathCauseCode

    roles = {case.case_id: case.role.value for case in design.cases}
    dimensions = {
        name: tuple(str(int(cause)) for cause in DeathCauseCode if cause is not DeathCauseCode.NONE)
        if name in {"deaths_by_cause", "death_fraction_by_cause"}
        else ("",)
        for name in names
    }
    rows: list[dict[str, object]] = []
    for cell, state in selection.cells.items():
        case, seed = cell
        data = selection.selected.get(cell)
        run = None if data is None else data.run
        observations: list[Observation] = (
            []
            if data is None
            else extract_metrics(
                data,
                design.analysis_policy,
                names,
                design.comparison_metrics.survival_fixed_age,
            )
        )
        values = {(item.name, item.dimension): item for item in observations}
        for name, keys in dimensions.items():
            for dimension in keys:
                metric = values.get((name, dimension))
                rows.append(
                    {
                        "experiment_id": str(design.experiment_id),
                        "case_id": case,
                        "role": roles[case],
                        "seed": seed,
                        "run_id": None if run is None else run.manifest.run_id,
                        "root_run_id": None
                        if run is None
                        else run.recovery.root_run_id
                        if run.recovery
                        else run.manifest.run_id,
                        "config_hash": None if run is None else run.manifest.config_hash,
                        "metric_name": name,
                        "dimension": dimension,
                        "metric_value": None if metric is None else metric.value,
                        "metric_quality": "UNAVAILABLE" if metric is None else metric.quality.value,
                        "availability_status": state
                        if metric is None
                        else "defined"
                        if metric.value is not None
                        else "right_censored"
                        if metric.reason == "right_censored"
                        else "unavailable",
                        "availability_reason": state if metric is None else metric.reason,
                        "coverage_start": None if data is None else data.coverage.start_step,
                        "coverage_end_exclusive": None
                        if data is None
                        else data.coverage.end_step_exclusive,
                    }
                )
    return pl.DataFrame(rows, schema=RUN_METRICS_SCHEMA).lazy()


def compare_experiment(
    design: ExperimentDesign,
    *,
    run_provider: RunProvider,
    comparison_policy: PairPolicy | None = None,
    choices: Mapping[Cell, str] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    bootstrap: BootstrapPolicy | None = None,
) -> ExperimentComparisonResult:
    """Return audit tables even when a strict contrast is blocked."""
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    resolved = resolve_design(design)
    policy = design.comparison_policy if comparison_policy is None else comparison_policy
    frozen = isinstance(run_provider, LocalRunProvider) and run_provider.matches_frozen_design(
        resolved
    )
    if isinstance(run_provider, LocalRunProvider) and run_provider.planned_execution and not frozen:
        raise ValueError("the requested design differs from the frozen execution plan")
    selection = select_comparison_runs(resolved, run_provider, choices=choices)
    planned = (
        frozen
        and policy is PairPolicy.STRICT
        and tolerance == DEFAULT_TOLERANCE
        and bootstrap is None
        and not choices
    )
    names = tuple(
        dict.fromkeys(
            (*design.comparison_metrics.names, "population_extinct", "time_to_extinction")
        )
    )
    metrics = _run_metrics(design, selection, names)
    baseline = metrics.filter(pl.col("case_id") == design.baseline_case_id)
    reports: list[ComparabilityReport] = []
    contrasts: list[pl.LazyFrame] = []
    warnings: set[str] = set()
    for case in resolved.differences:
        report = compare_compatibility(resolved, selection, case, policy)
        variant = metrics.filter(pl.col("case_id") == case)
        paired = pl.concat([compare_paired_metrics(baseline, variant, name) for name in names])
        paired = paired.with_columns(pl.lit(case).alias("variant_case_id"))
        eligible = pl.col("seed").is_in(report.paired_seeds)
        if report.comparison_eligible:
            undefined = (
                paired.filter(
                    eligible
                    & pl.col("metric_name").is_in(design.comparison_metrics.primary)
                    & pl.col("absolute_difference").is_null()
                )
                .select("metric_name")
                .unique()
                .sort("metric_name")
                .collect()["metric_name"]
                .to_list()
            )
            if undefined and policy is PairPolicy.STRICT:
                report = replace(
                    report,
                    comparison_eligible=False,
                    blocking_errors=(
                        *report.blocking_errors,
                        f"Incomplete primary metrics: {', '.join(undefined)}",
                    ),
                )
        delta_columns = ("absolute_difference", "relative_difference")
        contrasts.append(
            paired.with_columns(
                (eligible & pl.lit(report.comparison_eligible)).alias("pair_eligible"),
                *(
                    pl.when(eligible & pl.lit(report.comparison_eligible))
                    .then(pl.col(column))
                    .alias(column)
                    for column in delta_columns
                ),
            )
        )
        reports.append(report)
        warnings.update(report.warnings)
    if not planned:
        warnings.add("Exploratory analysis")
    if bootstrap is not None:
        warnings.add(
            "Percentile bootstrap intervals are exploratory and sensitive to limited seed counts"
        )
    differences = pl.concat(contrasts).sort("variant_case_id", *METRIC_KEYS, "seed")
    summaries = summarize_paired(differences, tolerance)
    if bootstrap is not None:
        summaries = summaries.join(
            bootstrap_summaries(differences, bootstrap),
            on=["variant_case_id", *METRIC_KEYS],
            validate="1:1",
        ).sort("variant_case_id", *METRIC_KEYS)
    unavailable = (
        metrics.filter(pl.col("metric_value").is_null()).group_by("metric_name").len().collect()
    )
    return ExperimentComparisonResult(
        str(uuid4()),
        COMPARISON_SCHEMA_VERSION,
        resolved,
        policy,
        "PLANNED" if planned else "EXPLORATORY",
        selection,
        tuple(reports),
        {
            "seed_coverage": _coverage_report(selection),
            "run_metrics": metrics,
            "case_summaries": summarize_case_metrics(metrics),
            "paired_differences": differences,
            "paired_summaries": summaries,
            **extinction_tables(metrics, differences, design.total_steps),
        },
        tuple(sorted(warnings)),
        dict(unavailable.iter_rows()),
        tolerance,
        bootstrap,
    )
