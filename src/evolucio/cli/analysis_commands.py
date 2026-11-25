"""Compact CLI views of the persisted-run analysis API."""

import json
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from evolucio.analysis.types import AnalysisPolicy, AnalysisWindow

analysis_app = typer.Typer(help="Analyze verified local run datasets.")

WindowSteps = Annotated[int, typer.Option("--window-steps", min=1)]
CohortWidth = Annotated[int, typer.Option("--cohort-width-steps", min=1)]
Fraction = Annotated[float, typer.Option("--bottleneck-fraction")]
StartStep = Annotated[int | None, typer.Option("--start-step", min=0)]
EndStep = Annotated[int | None, typer.Option("--end-step-exclusive", min=1)]


class OutputFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


def _window(start: int | None, end: int | None) -> AnalysisWindow | None:
    if start is None and end is None:
        return None
    if start is None or end is None:
        raise typer.BadParameter("provide both --start-step and --end-step-exclusive")
    return AnalysisWindow(start, end)


def _view(
    section: str | None,
    run_dir: Path,
    policy: AnalysisPolicy,
    window: AnalysisWindow | None,
    output: OutputFormat,
) -> None:
    from evolucio.analysis.energy import energy_by_window
    from evolucio.analysis.population import population_by_window
    from evolucio.analysis.summary import analyze_run
    from evolucio.analysis.survival import cohort_summary
    from evolucio.analysis.windows import validated
    from evolucio.persistence.loaders.run_data import open_run

    data = validated(open_run(run_dir))
    result = analyze_run(data, policy=policy, window=window)
    payload = asdict(result)
    if section is not None:
        payload = {
            name: value
            for name, value in payload.items()
            if not name.endswith("_summary")
            or name in (f"{section}_summary", "resource_summary" if section == "energy" else "")
        }
    preview = None
    if section == "population" and result.population_summary is not None:
        preview = population_by_window(data, policy.window_steps, window).select(
            "covered_start",
            "window_end_exclusive",
            "population_start",
            "population_end",
            "births_total",
            "deaths_total",
        )
    elif section == "energy":
        preview = energy_by_window(data, policy.window_steps, window).select(
            "covered_start",
            "window_end_exclusive",
            "metric_samples",
            "stock_statistics_quality",
            "gross_energy_cost_total",
            "resource_consumed_total",
        )
    elif section == "survival" and result.survival_summary is not None:
        preview = cohort_summary(data, policy.cohort_width_steps).select(
            "cohort_start",
            "is_founder",
            "agents_total",
            "deaths_observed",
            "right_censored_count",
        )
    if preview is not None:
        sample = preview.limit(6).collect()
        payload["preview"] = sample.head(5).to_dicts()
        payload["preview_truncated"] = sample.height > 5
    if output is OutputFormat.JSON:
        typer.echo(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))
        return
    coverage = result.coverage
    typer.echo(f"{result.run_id} ({result.run_status})")
    typer.echo(
        f"Steps [{coverage.start_step}, {coverage.end_step_exclusive}) | "
        f"metrics interval {result.metrics_interval}"
    )
    for name, value in payload.items():
        if name.endswith("_summary") and value is not None:
            heading = name.removesuffix("_summary")
            typer.echo(
                f"{heading} [EXACT]"
                if heading in ("population", "mortality", "survival")
                else heading
            )
            for label, field in value.items():
                if isinstance(field, dict) and "quality" in field:
                    typer.echo(f"  {label} = {field['value']} [{field['quality']}]")
                elif not isinstance(field, list):
                    typer.echo(f"  {label} = {field}")
                else:
                    typer.echo(f"  {label} = {json.dumps(field, ensure_ascii=False)}")
    if result.bottleneck_episode_count is not None and section in (None, "population"):
        typer.echo(f"bottleneck_episode_count = {result.bottleneck_episode_count}")
    if preview is not None:
        for row in payload["preview"]:
            typer.echo("  " + " | ".join(f"{name} = {value}" for name, value in row.items()))
        if payload["preview_truncated"]:
            typer.echo("  … export for the full table")
    for warning in result.warnings:
        typer.echo(f"Warning | {warning}")
    for name, reason in result.unavailable_metrics.items():
        typer.echo(f"UNAVAILABLE | {name} | {reason}")


Format = Annotated[OutputFormat, typer.Option("--format")]


@analysis_app.command()
def summary(
    run_dir: Path,
    window_steps: WindowSteps = 100,
    cohort_width_steps: CohortWidth = 100,
    bottleneck_fraction: Fraction = 0.1,
    start_step: StartStep = None,
    end_step_exclusive: EndStep = None,
    format: Format = OutputFormat.TEXT,
) -> None:
    """Summarize the verified run or an absolute step interval."""
    _view(
        None,
        run_dir,
        AnalysisPolicy(window_steps, cohort_width_steps, bottleneck_fraction),
        _window(start_step, end_step_exclusive),
        format,
    )


@analysis_app.command()
def population(
    run_dir: Path,
    window_steps: WindowSteps = 100,
    bottleneck_fraction: Fraction = 0.1,
    start_step: StartStep = None,
    end_step_exclusive: EndStep = None,
    format: Format = OutputFormat.TEXT,
) -> None:
    """Summarize demographic changes and bottlenecks."""
    _view(
        "population",
        run_dir,
        AnalysisPolicy(window_steps, bottleneck_fraction=bottleneck_fraction),
        _window(start_step, end_step_exclusive),
        format,
    )


@analysis_app.command()
def energy(
    run_dir: Path,
    window_steps: WindowSteps = 100,
    start_step: StartStep = None,
    end_step_exclusive: EndStep = None,
    format: Format = OutputFormat.TEXT,
) -> None:
    """Summarize energy and resource stocks and flows."""
    _view(
        "energy",
        run_dir,
        AnalysisPolicy(window_steps),
        _window(start_step, end_step_exclusive),
        format,
    )


@analysis_app.command()
def mortality(
    run_dir: Path,
    start_step: StartStep = None,
    end_step_exclusive: EndStep = None,
    format: Format = OutputFormat.TEXT,
) -> None:
    """Summarize mortality by cause and phase."""
    _view("mortality", run_dir, AnalysisPolicy(), _window(start_step, end_step_exclusive), format)


@analysis_app.command()
def survival(
    run_dir: Path,
    cohort_width_steps: CohortWidth = 100,
    format: Format = OutputFormat.TEXT,
) -> None:
    """Summarize survival with right censoring."""
    _view("survival", run_dir, AnalysisPolicy(cohort_width_steps=cohort_width_steps), None, format)


@analysis_app.command("export")
def export(
    run_dir: Path,
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    window_steps: WindowSteps = 100,
    cohort_width_steps: CohortWidth = 100,
    bottleneck_fraction: Fraction = 0.1,
    start_step: StartStep = None,
    end_step_exclusive: EndStep = None,
) -> None:
    """Write available analysis tables and source provenance."""
    from evolucio.analysis.export import export_analysis
    from evolucio.persistence.loaders.run_data import open_run

    typer.echo(
        export_analysis(
            open_run(run_dir),
            output_dir,
            policy=AnalysisPolicy(window_steps, cohort_width_steps, bottleneck_fraction),
            window=_window(start_step, end_step_exclusive),
        )
    )
