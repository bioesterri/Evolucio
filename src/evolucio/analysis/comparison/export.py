"""Write derived reports separately from immutable source runs."""

from collections.abc import Collection, Mapping
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl

from evolucio.analysis.types import ANALYSIS_SCHEMAS
from evolucio.experiments.design import plan_payload
from evolucio.persistence.local.artifacts import json_text, sha256_file
from evolucio.persistence.local.manifest import utc_now

from .compare import ExperimentComparisonResult
from .metrics import METRICS


def selection_table(result: ExperimentComparisonResult) -> pl.LazyFrame:
    selection = result.run_selection_report
    rows: list[dict[str, object]] = []
    for attempt in selection.attempts:
        data = (
            selection.selected.get((attempt.case_id, attempt.seed))
            if attempt.case_id is not None and attempt.seed is not None
            else None
        )
        selected = data is not None and data.run.manifest.run_id == attempt.run_id
        rows.append(
            {
                **asdict(attempt),
                "selected": selected,
                "reason": attempt.reason or (None if selected else "attempt not selected"),
            }
        )
    for (case, seed), state in selection.cells.items():
        rows.append(
            {
                "case_id": case,
                "seed": seed,
                "status": state,
                "selected": None,
                "reason": "planned cell status",
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "case_id": pl.String,
            "seed": pl.Int64,
            "run_id": pl.String,
            "root_run_id": pl.String,
            "config_hash": pl.String,
            "status": pl.String,
            "reason": pl.String,
            "selected": pl.Boolean,
        },
    ).lazy()


def render_report(result: ExperimentComparisonResult) -> str:
    design = result.experiment_design.design
    distributions = result.case_summaries.collect()
    coverage = result.tables["seed_coverage"].collect()
    return "\n".join(
        (
            f"# {design.name}",
            f"\n{result.interpretation} · {result.comparison_policy.value}",
            f"\n{design.hypothesis.question}",
            f"\nFactor estudiat · {design.hypothesis.factor}",
            f"\nCriteri declarat · {design.hypothesis.criteria}",
            *(
                f"- {name} · direcció esperada {direction}"
                for name, direction in design.hypothesis.expected_direction.items()
            ),
            f"\nBaseline · {result.baseline_case_id}\n",
            *(f"- {case.case_id} ({case.role.value}) {case.description}" for case in design.cases),
            "\n## Comparabilitat i cobertura\n",
            *(
                f"- {item.variant_case_id}: parelles {item.paired_seeds}, "
                f"absents {item.missing_seeds}, bloquejos {item.blocking_errors}."
                for item in result.comparability_reports
            ),
            *(
                f"- {item.variant_case_id} · {change.parameter_path} · "
                f"{change.baseline_value} → {change.variant_value} ({change.difference_category})"
                for item in result.comparability_reports
                for change in item.config_differences
            ),
            "",
            "| Cas | Previstes | Completes | Elegibles | Fallides | Aturades | Absents | Excl. |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *(
                "| " + " | ".join(str(value) for value in row) + " |"
                for row in coverage.iter_rows()
            ),
            "\n## Intents exclosos\n",
            *(
                f"- {row['case_id']} · {row['seed']} · {row['run_id']} · "
                f"{row['status']} · {row['reason']}"
                for row in selection_table(result)
                .filter(~pl.col("selected"))
                .collect()
                .iter_rows(named=True)
            ),
            "\n## Distribucions per cas\n",
            "Els valors per llavor i els motius d'absència són a la taula de mètriques per run.",
            "",
            "| Cas | Mètrica | n | Mitjana | Mediana | Q25 | Q75 | Desviació mostral |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
            *(
                f"| {row['case_id']} | {row['metric_name']} {row['dimension']} "
                f"| {row['n_available']}/{row['n_expected']} | {row['mean']} | {row['median']} "
                f"| {row['q25']} | {row['q75']} | {row['sample_std']} |"
                for row in distributions.iter_rows(named=True)
            ),
            "\n## Diferències aparellades i consistència\n",
            *(
                f"- {row['variant_case_id']} · {row['metric_name']} {row['dimension']} · "
                f"n={row['n_pairs']}, mitjana={row['mean_difference']}, "
                f"mediana={row['median_difference']}, "
                f"signes (+/-/=)={row['positive_difference_count']}/"
                f"{row['negative_difference_count']}/{row['equal_difference_count']}"
                for row in result.paired_difference_summaries.collect().iter_rows(named=True)
            ),
            "\n## Limitacions\n",
            *(f"- {warning}" for warning in result.warnings),
            "- Inferir adaptació a partir dels efectes observats requereix evidència "
            "addicional sobre herència i selecció.",
            "- El registre distingeix les extincions observades i els seguiments censurats.",
            "",
        )
    )


def export_comparison(
    result: ExperimentComparisonResult,
    output_dir: Path,
    *,
    tables: Collection[str] | None = None,
    temporal_tables: Mapping[str, pl.LazyFrame] | None = None,
    temporal_context: Mapping[str, object] | None = None,
) -> Path:
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(target)
    sources = [analysis.run for analysis in result.run_selection_report.selected.values()]
    if any(target.is_relative_to(run.path) for run in sources) or any(
        (parent / "manifest.json").is_file() for parent in target.parents
    ):
        raise ValueError("comparison output must be outside source run directories")
    available = {**result.tables, "run_selection": selection_table(result)}
    for name, table in (temporal_tables or {}).items():
        if name not in {"time_series_summary", "paired_time_series", "differential_change"}:
            raise ValueError(f"unknown temporal artifact: {name}")
        available[name] = table
    chosen = tuple(available) if tables is None else tuple(tables)
    if set(chosen) - available.keys():
        raise ValueError("requested comparison artifacts unavailable")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as temporary:
        stage = Path(temporary) / "comparison"
        stage.mkdir()
        for name in sorted(set(chosen)):
            available[name].sink_parquet(stage / f"{name}.parquet")
        context = {
            "comparison_id": result.comparison_id,
            "experiment_id": result.experiment_id,
            "schema_version": result.schema_version,
            "created_at_utc": utc_now(),
            "interpretation": result.interpretation,
            "comparison_policy": result.comparison_policy,
            "analysis_policy": asdict(result.experiment_design.design.analysis_policy),
            "analysis_schema_versions": {**ANALYSIS_SCHEMAS, "lineage_genetic_analysis": 1},
            "numpy_version": np.__version__,
            "polars_version": pl.__version__,
            "tolerance": result.tolerance,
            "temporal_analysis": temporal_context,
            "bootstrap": None if result.bootstrap is None else asdict(result.bootstrap),
            "sources": [
                {
                    "run_id": run.manifest.run_id,
                    "config_hash": run.manifest.config_hash,
                    "recovery": asdict(run.recovery) if run.recovery is not None else None,
                    "manifest_sha256": sha256_file(run.path / "manifest.json"),
                    "artifacts": [asdict(item) for item in run.manifest.artifacts],
                }
                for run in sources
            ],
            "coverage": result.tables["seed_coverage"].collect().to_dicts(),
            "warnings": result.warnings,
            "unavailable_metrics": result.unavailable_metrics,
            "artifacts": sorted(f"{name}.parquet" for name in set(chosen)),
        }
        names = result.tables["run_metrics"].select("metric_name").unique().collect().to_series()
        for name, payload in (
            ("manifest", context),
            ("experiment_design", plan_payload(result.experiment_design)),
            ("comparability_report", [asdict(report) for report in result.comparability_reports]),
            ("metric_registry", {name: asdict(METRICS[name]) for name in sorted(names)}),
        ):
            (stage / f"{name}.json").write_text(json_text(payload), encoding="utf-8")
        (stage / "report.md").write_text(render_report(result), encoding="utf-8")
        stage.rename(target)
    return target
