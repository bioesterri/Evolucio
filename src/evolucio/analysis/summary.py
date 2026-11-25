"""Compact scientific summaries over a verified persisted trajectory."""

from dataclasses import dataclass, fields
from typing import cast

from evolucio.persistence.loaders.run_data import RunData

from .energy import EnergySummary, ResourceSummary, energy_and_resources
from .mortality import death_age_distribution, mortality_by_cause, mortality_by_phase
from .population import PopulationSummary, bottlenecks, population_summary
from .survival import SurvivalSummary, survival_summary
from .types import (
    ANALYSIS_SCHEMA_VERSION,
    ANALYSIS_SCHEMAS,
    DEFAULT_ANALYSIS_POLICY,
    AnalysisPolicy,
    AnalysisUnavailableError,
    AnalysisWindow,
    Coverage,
    Metric,
    Quality,
)
from .windows import AnalysisRun, count, validated


@dataclass(frozen=True, slots=True)
class MortalitySummary:
    deaths_total: int
    by_cause: list[dict[str, object]]
    by_phase: list[dict[str, object]]
    observed_death_age: dict[str, object]


@dataclass(frozen=True, slots=True)
class RunAnalysisResult:
    run_id: str
    root_run_id: str
    source_run_id: str | None
    recovery_depth: int
    run_status: str
    analysis_schema_version: int
    analysis_schemas: dict[str, int]
    analysis_policy: AnalysisPolicy
    coverage: Coverage
    metrics_interval: int
    population_summary: PopulationSummary | None
    bottleneck_episode_count: int | None
    energy_summary: EnergySummary
    resource_summary: ResourceSummary
    mortality_summary: MortalitySummary
    survival_summary: SurvivalSummary | None
    warnings: tuple[str, ...]
    unavailable_metrics: dict[str, str]


def analyze_run(
    run: RunData | AnalysisRun,
    *,
    policy: AnalysisPolicy = DEFAULT_ANALYSIS_POLICY,
    window: AnalysisWindow | None = None,
) -> RunAnalysisResult:
    data = validated(run)
    coverage = data.bounds(window)
    warnings = [issue.message for issue in data.report.issues]
    unavailable: dict[str, str] = {}
    population = population_summary(data, window) if data.population_available else None
    episodes = None
    if population is None:
        unavailable["population_summary"] = "validated trajectory origin unavailable"
    if data.population_available and data.run.config.population.initial_agents > 0:
        episodes = int(bottlenecks(data, policy, window).select(count()).collect().item())
    else:
        unavailable["bottleneck_episode_count"] = (
            "positive validated initial founder population required"
        )

    energy, resources = energy_and_resources(data, window, population)
    for section, metrics in (("energy", energy), ("resources", resources)):
        for field in fields(metrics):
            metric = cast(Metric, getattr(metrics, field.name))
            if metric.quality is Quality.UNAVAILABLE:
                unavailable[f"{section}.{field.name}"] = metric.limitations or "unavailable"

    causes = mortality_by_cause(data, window).collect()
    mortality = MortalitySummary(
        int(causes["death_count"].sum()),
        causes.to_dicts(),
        mortality_by_phase(data, window).collect().to_dicts(),
        death_age_distribution(data, window).collect().row(0, named=True),
    )
    survival = None
    if coverage == data.coverage:
        try:
            survival = survival_summary(data, policy)
        except AnalysisUnavailableError as exc:
            unavailable["survival_summary"] = str(exc)
    else:
        unavailable["survival_summary"] = (
            "survival requires full follow-up from each observed origin"
        )
    if survival is not None and survival.median_survival_age is None:
        unavailable["median_survival_age"] = "the observed survival curve did not reach 0.5"

    provenance = data.run.recovery
    run_id = data.run.manifest.run_id
    if data.run.manifest.status.value != "completed":
        warnings.append("Analysis covers the committed terminal prefix.")
    return RunAnalysisResult(
        run_id,
        provenance.root_run_id if provenance else run_id,
        provenance.source_run_id if provenance else None,
        provenance.recovery_depth if provenance else 0,
        data.run.manifest.status.value,
        ANALYSIS_SCHEMA_VERSION,
        dict(ANALYSIS_SCHEMAS),
        policy,
        coverage,
        data.metrics_interval,
        population,
        episodes,
        energy,
        resources,
        mortality,
        survival,
        tuple(warnings),
        unavailable,
    )
