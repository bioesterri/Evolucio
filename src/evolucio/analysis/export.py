"""Explicit export to a separate, newly created derived directory."""

from dataclasses import asdict
from pathlib import Path
from shutil import rmtree

import polars as pl

from evolucio.persistence.loaders.run_data import RunData
from evolucio.persistence.local.artifacts import atomic_write_text, json_text

from .energy import energy_by_window
from .mortality import mortality_by_cause, mortality_by_window
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

    index = source.parquet
    assert index is not None
    payload = asdict(result)
    payload.update(
        source_run_id=result.run_id,
        recovery_source_run_id=result.source_run_id,
        source_config_hash=source.manifest.config_hash,
        source_dataset_hashes={
            **{a.relative_path: a.sha256 for a in source.manifest.artifacts},
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
    target.mkdir(parents=True, exist_ok=False)
    try:
        for name, (schema, frame) in tables.items():
            frame.sink_parquet(
                target / f"{name}.parquet",
                metadata={
                    "schema_name": schema,
                    "schema_version": str(result.analysis_schemas[schema]),
                    "run_id": result.run_id,
                },
            )
        atomic_write_text(target / "analysis_summary.json", json_text(payload))
    except BaseException:
        rmtree(target)
        raise
    return target
