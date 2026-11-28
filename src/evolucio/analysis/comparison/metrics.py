"""Run-level metric registry and adapters to PR-43/44 definitions."""

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Literal, cast

import polars as pl

from evolucio.analysis.genealogy import genealogy_nodes
from evolucio.analysis.genetic_diversity import (
    genetic_diversity_at_checkpoint,
    genetic_diversity_series,
)
from evolucio.analysis.genome_distance import founder_genome_divergence
from evolucio.analysis.lineage_analysis import analyze_lineages
from evolucio.analysis.population import bottlenecks
from evolucio.analysis.summary import analyze_run
from evolucio.analysis.survival import survival_curve
from evolucio.analysis.types import AnalysisPolicy, AnalysisUnavailableError, Metric, Quality
from evolucio.analysis.windows import AnalysisRun

if TYPE_CHECKING:
    from evolucio.experiments.design import ExperimentDesign


@dataclass(frozen=True, slots=True)
class MetricSpecification:
    metric_name: str
    definition: str
    unit: str
    source: str
    aggregation: str
    availability_requirements: str
    ratio_scale: bool = True
    schema_version: int = 1
    direction: Literal["no_direction"] = "no_direction"


METRICS: dict[str, MetricSpecification] = {}


def validate_metrics(design: "ExperimentDesign") -> None:
    metrics = design.comparison_metrics
    if unknown := set(metrics.names) - METRICS.keys():
        raise ValueError(f"unknown comparison metrics: {sorted(unknown)}")
    if "survival_at_fixed_age" in metrics.names and metrics.survival_fixed_age is None:
        raise ValueError("survival_at_fixed_age requires a declared survival_fixed_age")


def _register(
    name: str,
    definition: str,
    unit: str,
    source: str,
    aggregation: str,
    requires: str,
    *,
    ratio_scale: bool = True,
) -> None:
    METRICS[name] = MetricSpecification(
        name,
        definition,
        unit,
        source,
        aggregation,
        requires,
        ratio_scale,
    )


for _name, _definition, _unit in (
    ("initial_population", "Live population at the initial state", "agents"),
    ("final_population", "Live population at the final state", "agents"),
    ("mean_population", "Mean post-transition live population", "agents"),
    ("median_population", "Median post-transition live population", "agents"),
    ("minimum_population", "Minimum post-transition live population", "agents"),
    ("maximum_population", "Maximum post-transition live population", "agents"),
    ("population_cv", "Population standard deviation divided by mean", "ratio"),
    ("births_total", "Committed births", "events"),
    ("deaths_total", "Committed deaths", "events"),
    ("population_extinct", "Population reached zero within the horizon", "indicator"),
    ("time_to_extinction", "Elapsed steps to observed extinction", "steps"),
    ("bottleneck_episode_count", "Observed below-threshold episodes", "episodes"),
    ("bottleneck_observed_duration", "Total observed steps within bottlenecks", "steps"),
):
    _register(
        _name,
        _definition,
        _unit,
        "PR-43 population",
        "full horizon",
        "verified population and events",
        ratio_scale=_name not in {"population_extinct", "time_to_extinction"},
    )

ENERGY_FIELDS = {
    "resource_consumed_total": "resources",
    "initial_resources": "resources",
    "final_resources": "resources",
    "observed_mean_resources": "resources",
    "feeding_energy_gained_total": "energy",
    "gross_energy_cost_total": "energy",
    "net_reproductive_energy_cost_total": "energy",
    "births_per_resource_consumed": "births/resource",
    "agent_steps_per_energy_cost": "agent-steps/energy",
    "energy_balance_residual": "energy",
}
RESOURCE_FIELDS = frozenset(
    {
        "resource_consumed_total",
        "initial_resources",
        "final_resources",
        "observed_mean_resources",
    }
)
for _name, _unit in ENERGY_FIELDS.items():
    _register(
        _name,
        _name.replace("_", " "),
        _unit,
        "PR-43 energy/resources",
        "full horizon",
        "exact PR-43 metric coverage",
        ratio_scale=_name not in {"energy_balance_residual", "net_reproductive_energy_cost_total"},
    )

for _name, _definition, _unit in (
    ("observed_death_age_mean", "Mean age of observed deaths", "steps"),
    ("observed_death_age_median", "Median age of observed deaths", "steps"),
    ("median_survival_age", "First age with Kaplan-Meier survival at or below 0.5", "steps"),
    ("survival_at_fixed_age", "Kaplan-Meier survival at the declared age", "fraction"),
):
    _register(
        _name,
        _definition,
        _unit,
        "PR-43 mortality/survival",
        "full horizon",
        "verified events and identifiable statistic",
        ratio_scale=_unit != "fraction",
    )

for _name, _unit in (
    ("direct_offspring_mean", "offspring/agent"),
    ("direct_offspring_median", "offspring/agent"),
    ("proportion_without_offspring_observed", "fraction"),
    ("total_descendants", "agents"),
    ("maximum_generation", "generation"),
    ("active_lineage_count", "lineages"),
    ("extinct_lineage_count", "lineages"),
    ("dominant_lineage_fraction", "fraction"),
    ("shannon_entropy", "nats"),
    ("simpson_diversity", "index"),
    ("effective_lineages_shannon", "lineages"),
    ("effective_lineages_simpson", "lineages"),
    ("mutation_effective_total", "parameters"),
    ("mean_effective_mutations_per_birth", "parameters/birth"),
):
    _register(
        _name,
        _name.replace("_", " "),
        _unit,
        "PR-44 genealogy",
        "full horizon or terminal census",
        "verified founder census and events",
        ratio_scale=_unit not in {"fraction", "nats", "index", "generation"},
    )

GENETIC_FIELDS = frozenset(
    {
        "genetic_parameter_variance",
        "pairwise_rms_distance_per_parameter",
        "within_lineage_variance",
        "between_lineage_variance",
        "founder_genome_divergence",
    }
)
for _name in sorted(GENETIC_FIELDS):
    _register(
        _name,
        _name.replace("_", " "),
        "parameter²" if "variance" in _name else "parameter",
        "PR-44 genetic diversity",
        "mean RMS across retained lineage representatives"
        if _name == "founder_genome_divergence"
        else "terminal live population",
        "terminal checkpoint, exact final sample, or verified retained genomes as applicable",
    )

for _name in ("deaths_by_cause", "death_fraction_by_cause"):
    _register(
        _name,
        _name.replace("_", " "),
        "events" if _name == "deaths_by_cause" else "fraction",
        "PR-43 mortality",
        "full horizon by cause",
        "verified deaths",
        ratio_scale=_name == "deaths_by_cause",
    )


@dataclass(frozen=True, slots=True)
class Observation:
    name: str
    value: float | None
    quality: Quality
    reason: str | None = None
    dimension: str = ""


def _observation(name: str, value: object, dimension: str = "") -> Observation:
    numeric = float(value) if isinstance(value, (int, float)) and isfinite(value) else None
    return Observation(
        name,
        numeric,
        Quality.EXACT if numeric is not None else Quality.UNAVAILABLE,
        None if numeric is not None else "statistic not defined or source unavailable",
        dimension,
    )


def _section(values: dict[str, object], name: str) -> dict[str, object]:
    return cast(dict[str, object], values.get(name) or {})


def _lineage_metrics(data: AnalysisRun, policy: AnalysisPolicy) -> dict[str, object]:
    result = analyze_lineages(data, policy=policy).summaries
    mutations = _section(result, "mutations")
    offspring = _section(result, "offspring")
    diversity = _section(result, "diversity")
    lineages = _section(result, "lineages")
    births = cast(int | None, mutations.get("births_total"))
    effective = cast(int | None, mutations.get("mutation_effective_total"))
    return {
        **diversity,
        **mutations,
        "direct_offspring_mean": offspring.get("mean_direct_offspring"),
        "direct_offspring_median": offspring.get("median_direct_offspring"),
        "proportion_without_offspring_observed": offspring.get(
            "proportion_without_offspring_observed"
        ),
        "total_descendants": births,
        "maximum_generation": genealogy_nodes(data)
        .select(pl.col("generation").max())
        .collect()
        .item(),
        "active_lineage_count": lineages.get("lineages_alive_at_prefix_end"),
        "extinct_lineage_count": lineages.get("lineages_extinct"),
        "mean_effective_mutations_per_birth": effective / births
        if births and effective is not None
        else None,
    }


def _genetic_metrics(data: AnalysisRun, names: set[str]) -> dict[str, object]:
    end = data.coverage.end_step_exclusive
    values: dict[str, object] = {}
    if names & {"genetic_parameter_variance", "pairwise_rms_distance_per_parameter"}:
        sample = (
            genetic_diversity_series(data)
            .filter((pl.col("step") == end - 1) & (pl.col("population") > 0))
            .collect()
        )
        if sample.height:
            values.update(sample.row(0, named=True))
    if names & {"within_lineage_variance", "between_lineage_variance"}:
        try:
            checkpoint = genetic_diversity_at_checkpoint(data, end)
        except (AnalysisUnavailableError, FileNotFoundError):
            pass
        else:
            values.update(
                within_lineage_variance=checkpoint.within_lineage_variance,
                between_lineage_variance=checkpoint.between_lineage_variance,
            )
    if "founder_genome_divergence" in names:
        values["founder_genome_divergence"] = (
            founder_genome_divergence(data)
            .select(pl.col("rms_parameter_difference").mean())
            .collect()
            .item()
        )
    return values


def extract_metrics(
    data: AnalysisRun,
    policy: AnalysisPolicy,
    names: tuple[str, ...],
    fixed_age: int | None,
) -> list[Observation]:
    summary = analyze_run(data, policy=policy)
    population = summary.population_summary
    values: dict[str, object] = {
        "bottleneck_episode_count": summary.bottleneck_episode_count,
        "observed_death_age_mean": summary.mortality_summary.observed_death_age["mean"],
        "observed_death_age_median": summary.mortality_summary.observed_death_age["median"],
        "median_survival_age": None
        if summary.survival_summary is None
        else summary.survival_summary.median_survival_age,
    }
    if "bottleneck_observed_duration" in names and summary.bottleneck_episode_count is not None:
        values["bottleneck_observed_duration"] = (
            bottlenecks(data, policy).select(pl.col("duration_observed").sum()).collect().item()
        )
    if "survival_at_fixed_age" in names and fixed_age is not None and summary.survival_summary:
        curve = survival_curve(data, cohort_width_steps=policy.cohort_width_steps).collect()
        if curve.height and (
            fixed_age <= cast(int, curve["age"].max()) or curve["survival_probability"].last() == 0
        ):
            values["survival_at_fixed_age"] = (
                curve.filter(pl.col("age") <= fixed_age)
                .select(pl.col("survival_probability").last())
                .item()
            )
            if values["survival_at_fixed_age"] is None:
                values["survival_at_fixed_age"] = 1.0
    if any(METRICS[name].source == "PR-44 genealogy" for name in names):
        values.update(_lineage_metrics(data, policy))
    values.update(_genetic_metrics(data, set(names) & GENETIC_FIELDS))

    observations: list[Observation] = []
    for name in names:
        if name in ENERGY_FIELDS:
            metric = cast(
                Metric,
                getattr(
                    summary.energy_summary
                    if name not in RESOURCE_FIELDS
                    else summary.resource_summary,
                    name,
                ),
            )
            if metric.quality is not Quality.EXACT or metric.coverage != data.coverage:
                observations.append(
                    Observation(
                        name, None, metric.quality, metric.limitations or "exact horizon required"
                    )
                )
            else:
                observations.append(_observation(name, metric.value))
        elif name in {"deaths_by_cause", "death_fraction_by_cause"}:
            for cause in summary.mortality_summary.by_cause:
                column = "death_count" if name == "deaths_by_cause" else "death_fraction"
                observations.append(_observation(name, cause[column], str(cause["cause_code"])))
        elif (
            name == "time_to_extinction"
            and population is not None
            and not population.population_extinct
        ):
            observations.append(Observation(name, None, Quality.UNAVAILABLE, "right_censored"))
        else:
            observations.append(
                _observation(name, values.get(name, getattr(population, name, None)))
            )
    return observations
