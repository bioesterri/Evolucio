"""Explicit export to a separate, newly created derived directory."""

from dataclasses import asdict
from pathlib import Path
from shutil import rmtree

import polars as pl

from evolucio.persistence.loaders.run_data import RunData
from evolucio.persistence.local.artifacts import atomic_write_text, json_text

from .energy import energy_by_window
from .genetic_diversity import genetic_diversity_series
from .genome_distance import founder_genome_divergence
from .lineage_analysis import analyze_genetic_diversity, analyze_lineages
from .lineage_diversity import lineage_diversity_series
from .lineage_dynamics import lineage_extinction_summary, lineage_population_series
from .mortality import mortality_by_cause, mortality_by_window
from .mutation_analysis import mutation_summary_by_lineage
from .offspring import generation_summary, offspring_distribution
from .population import population_by_window
from .summary import analyze_run
from .survival import cohort_summary, survival_curve, survival_observations
from .types import DEFAULT_ANALYSIS_POLICY, AnalysisPolicy, AnalysisWindow
from .windows import AnalysisRun, validated


def export_analysis(
    run: RunData | AnalysisRun,
    output_dir: Path,
    *,
    policy: AnalysisPolicy = DEFAULT_ANALYSIS_POLICY,
    window: AnalysisWindow | None = None,
    include_lineages: bool = False,
) -> Path:
    data = validated(run)
    source = data.run
    target = output_dir.resolve()
    if target.is_relative_to(source.path):
        raise ValueError("analysis output must be outside the source run")
    result = analyze_run(data, policy=policy, window=window)
    tables: dict[str, tuple[str, pl.LazyFrame]] = {
        "mortality_by_cause": ("mortality_analysis", mortality_by_cause(data, window)),
        "energy_by_window": (
            "energy_analysis",
            energy_by_window(data, policy.window_steps, window),
        ),
        "mortality_by_window": (
            "mortality_analysis",
            mortality_by_window(data, policy.window_steps, window),
        ),
    }
    if data.population_available:
        tables["population_by_window"] = (
            "population_analysis",
            population_by_window(data, policy.window_steps, window),
        )
    if result.survival_summary is not None:
        width = policy.cohort_width_steps
        tables.update(
            survival_observations=("survival_analysis", survival_observations(data, width)),
            survival_curve=("survival_analysis", survival_curve(data, cohort_width_steps=width)),
            cohort_summary=("survival_analysis", cohort_summary(data, width)),
        )

    lineage_payload = (
        _add_lineage_tables(data, tables, policy, window) if include_lineages else None
    )

    index = source.parquet
    assert index is not None
    payload = asdict(result)
    payload.update(
        source_run_id=result.run_id,
        recovery_source_run_id=result.source_run_id,
        source_config_hash=source.manifest.config_hash,
        source_dataset_hashes={
            **{a.relative_path: a.sha256 for a in source.manifest.artifacts},
            **(
                {
                    record.relative_path: record.artifact_sha256
                    for record in source.list_highlighted_genomes()
                }
                if include_lineages
                else {}
            ),
            **{
                part.relative_path: part.sha256
                for ds in index.datasets.values()
                for part in ds.parts
            },
            **{
                record.relative_path: record.sha256
                for record in (*source.list_checkpoints(), *source.list_snapshots())
            },
        },
        exported_tables=sorted(tables),
    )
    analysis_schemas = dict(result.analysis_schemas)
    if lineage_payload is not None:
        analysis_schemas.update(LINEAGE_SCHEMAS)
        lineage_payload.update(
            source_config_hash=payload["source_config_hash"],
            source_dataset_hashes=payload["source_dataset_hashes"],
        )
    payload["analysis_schemas"] = analysis_schemas
    target.mkdir(parents=True, exist_ok=False)
    try:
        for name, (schema, frame) in tables.items():
            frame.sink_parquet(
                target / f"{name}.parquet",
                metadata={
                    "schema_name": schema,
                    "schema_version": str(analysis_schemas[schema]),
                    "run_id": result.run_id,
                },
            )
        atomic_write_text(target / "analysis_summary.json", json_text(payload))
        if lineage_payload is not None:
            atomic_write_text(target / "lineage_analysis_summary.json", json_text(lineage_payload))
    except BaseException:
        rmtree(target)
        raise
    return target


LINEAGE_SCHEMAS = {"lineage_analysis": 1, "genetic_diversity_analysis": 1}


def _add_lineage_tables(
    data: AnalysisRun,
    tables: dict[str, tuple[str, pl.LazyFrame]],
    policy: AnalysisPolicy,
    window: AnalysisWindow | None,
) -> dict[str, object]:
    lineage = analyze_lineages(data, policy=policy, window=window)
    genetics = analyze_genetic_diversity(data, policy=policy, window=window)
    if data.population_available:
        tables.update(
            lineage_population_series=(
                "lineage_analysis",
                lineage_population_series(data, window=window),
            ),
            lineage_diversity_series=("lineage_analysis", lineage_diversity_series(data, window)),
            mutation_summary_by_lineage=(
                "lineage_analysis",
                mutation_summary_by_lineage(data, window),
            ),
        )
        if lineage.coverage == data.coverage:
            tables.update(
                offspring_distribution=("lineage_analysis", offspring_distribution(data).histogram),
                generation_summary=("lineage_analysis", generation_summary(data)),
                lineage_extinction_summary=("lineage_analysis", lineage_extinction_summary(data)),
            )
            divergence = founder_genome_divergence(data)
            if divergence.limit(1).collect().height:
                tables["founder_genome_divergence"] = ("genetic_diversity_analysis", divergence)
    if genetics.summaries["metric_samples"]:
        tables["genetic_diversity_series"] = (
            "genetic_diversity_analysis",
            genetic_diversity_series(data, window),
        )

    return {
        **asdict(lineage),
        "genetic_diversity": asdict(genetics),
        "analysis_schemas": LINEAGE_SCHEMAS,
        "exported_tables": [
            name for name, (schema, _) in tables.items() if schema in LINEAGE_SCHEMAS
        ],
    }
