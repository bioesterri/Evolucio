"""Compact results for verified lineage and genetic analyses."""

from dataclasses import dataclass
from typing import TypedDict, cast

import polars as pl

from evolucio.persistence.loaders.run_data import RunData

from .genetic_diversity import genetic_diversity_series
from .lineage_diversity import lineage_diversity_series
from .lineage_dynamics import lineage_extinction_summary
from .mutation_analysis import mutation_summary_by_lineage
from .offspring import offspring_distribution
from .types import DEFAULT_ANALYSIS_POLICY, AnalysisPolicy, AnalysisWindow, Coverage, Quality
from .windows import AnalysisRun, count, validated


@dataclass(frozen=True, slots=True)
class LineageAnalysisResult:
    schema_version: int
    run_id: str
    root_run_id: str
    source_run_id: str | None
    run_status: str
    recovery_depth: int
    coverage: Coverage
    analysis_policy: AnalysisPolicy
    summaries: dict[str, object]
    warnings: tuple[str, ...]
    unavailable_metrics: dict[str, str]


@dataclass(frozen=True, slots=True)
class GeneticDiversityAnalysisResult(LineageAnalysisResult):
    """Compact summary of observed genetic variance and its coverage."""


class _ResultContext(TypedDict):
    schema_version: int
    run_id: str
    root_run_id: str
    source_run_id: str | None
    run_status: str
    recovery_depth: int
    coverage: Coverage
    analysis_policy: AnalysisPolicy
    warnings: tuple[str, ...]


def _context(data: AnalysisRun, policy: AnalysisPolicy, coverage: Coverage) -> _ResultContext:
    source = data.run.recovery
    run_id = data.run.manifest.run_id
    warnings = [issue.message for issue in data.report.issues]
    if data.run.manifest.status.value != "completed":
        warnings.append("Analysis covers the committed terminal prefix.")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "root_run_id": source.root_run_id if source else run_id,
        "source_run_id": source.source_run_id if source else None,
        "run_status": data.run.manifest.status.value,
        "recovery_depth": source.recovery_depth if source else 0,
        "coverage": coverage,
        "analysis_policy": policy,
        "warnings": tuple(warnings),
    }


def analyze_lineages(
    run: RunData | AnalysisRun,
    *,
    policy: AnalysisPolicy = DEFAULT_ANALYSIS_POLICY,
    window: AnalysisWindow | None = None,
) -> LineageAnalysisResult:
    data = validated(run)
    coverage = data.bounds(window)
    summaries: dict[str, object] = {}
    unavailable: dict[str, str] = {}
    if not data.population_available:
        unavailable["lineages"] = "validated founder census and genealogy required"
    else:
        series = lineage_diversity_series(data, window)
        summaries["diversity"] = next(iter(series.tail(1).collect().to_dicts()), None)
        summaries["quality"] = Quality.EXACT
        summaries["mutations"] = (
            mutation_summary_by_lineage(data, window)
            .select(
                pl.col(
                    "births_total",
                    "mutation_selected_total",
                    "mutation_effective_total",
                    "mutation_abs_delta_total",
                ).sum()
            )
            .collect()
            .row(0, named=True)
        )
        if coverage == data.coverage:
            summaries["offspring"] = offspring_distribution(data).summary
            summaries["lineages"] = (
                lineage_extinction_summary(data)
                .select(
                    count().alias("lineages_total"),
                    pl.col("extinction_observed").sum().alias("lineages_extinct"),
                    (pl.col("alive_final") > 0).sum().alias("lineages_alive_at_prefix_end"),
                )
                .collect()
                .row(0, named=True)
            )
        else:
            unavailable["lifecycle_summaries"] = "lifecycle summaries use the full verified prefix"
    return LineageAnalysisResult(
        **_context(data, policy, coverage),
        summaries=summaries,
        unavailable_metrics=unavailable,
    )


def analyze_genetic_diversity(
    run: RunData | AnalysisRun,
    *,
    policy: AnalysisPolicy = DEFAULT_ANALYSIS_POLICY,
    window: AnalysisWindow | None = None,
) -> GeneticDiversityAnalysisResult:
    data = validated(run)
    coverage = data.bounds(window)
    series = genetic_diversity_series(data, window)
    count_samples = cast(int, series.select(count()).collect().item())
    summaries: dict[str, object] = {
        "metric_samples": count_samples,
        "metrics_interval": data.metrics_interval,
        "scope": "full_live_population_at_each_persisted_metric_step",
        "source": "metrics Parquet",
        "temporal_quality": (
            Quality.UNAVAILABLE
            if count_samples == 0
            else Quality.EXACT
            if count_samples == coverage.end_step_exclusive - coverage.start_step
            else Quality.SAMPLED
        ),
        "last_sample": next(iter(series.tail(1).collect().to_dicts()), None),
    }
    unavailable: dict[str, str] = {}
    if count_samples == 0:
        unavailable["genetic_parameter_variance"] = "no persisted metrics in the analysis window"
    if not data.run.list_highlighted_genomes():
        unavailable["retained_genome_distances"] = "no highlighted genomes are indexed"
    if not data.run.list_checkpoints():
        unavailable["within_and_between_lineage_variance"] = "no full checkpoint is indexed"
    return GeneticDiversityAnalysisResult(
        **_context(data, policy, coverage),
        summaries=summaries,
        unavailable_metrics=unavailable,
    )
