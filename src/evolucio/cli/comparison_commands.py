"""CLI entry points for planned execution and experimental comparison."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from .analysis_commands import analysis_app
from .database_commands import experiment_app


@experiment_app.command("validate-design")
def validate_design(path: Path) -> None:
    """Show the validated experimental matrix without starting runs."""
    from evolucio.analysis.comparison.compare import validate_metrics
    from evolucio.experiments.design import load_design, plan_payload, resolve_design

    design = load_design(path)
    validate_metrics(design)
    resolved = resolve_design(design)
    typer.echo(
        json.dumps(
            {
                **plan_payload(resolved),
                "blocking_errors": resolved.blocking_errors,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if any(resolved.blocking_errors.values()):
        raise typer.Exit(2)


@experiment_app.command("execute-design")
def execute(
    path: Path,
    execution_dir: Annotated[Path | None, typer.Option("--execution-dir")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Freeze the plan and execute it via the existing batch runtime."""
    from evolucio.analysis.comparison.compare import validate_metrics
    from evolucio.experiments.design import execute_design, load_design, resolve_design

    if dry_run:
        validate_design(path)
        return
    design = load_design(path)
    validate_metrics(design)
    target = execution_dir or Path("experiments") / str(design.experiment_id) / str(
        design.design_version
    )
    outcome = execute_design(resolve_design(design), target)
    typer.echo(f"{target} · {outcome.completed_run_count}/{outcome.requested_run_count} completats")
    if outcome.stopped_run_count or outcome.not_started_run_count:
        raise typer.Exit(130)
    if outcome.failed_run_count:
        raise typer.Exit(1)


@analysis_app.command("compare")
def compare(
    path: Path,
    execution_dir: Annotated[Path | None, typer.Option("--execution-dir")] = None,
    runs_root: Annotated[Path | None, typer.Option("--runs-root")] = None,
    registered: Annotated[bool, typer.Option("--registered")] = False,
    allow_available_pairs: Annotated[bool, typer.Option("--allow-available-pairs")] = False,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    select_run: Annotated[
        list[str] | None, typer.Option("--select-run", help="CASE:SEED=RUN_ID")
    ] = None,
    bootstrap_seed: Annotated[int | None, typer.Option("--bootstrap-seed")] = None,
    temporal_metric: Annotated[str | None, typer.Option("--temporal-metric")] = None,
    environment_control: Annotated[str | None, typer.Option("--environment-control")] = None,
    common_steps: Annotated[bool, typer.Option("--common-steps")] = False,
) -> None:
    """Verify persisted trajectories and compare matched seed blocks."""
    from evolucio.analysis.comparison import (
        AlignmentPolicy,
        BootstrapPolicy,
        LocalRunProvider,
        RegisteredRunProvider,
        compare_experiment,
        export_comparison,
    )
    from evolucio.analysis.comparison.export import render_report
    from evolucio.analysis.comparison.temporal import (
        aligned_series,
        differential_change,
        environment_windows,
        paired_time_series,
        summarize_time_series,
    )
    from evolucio.analysis.types import AnalysisWindow, Coverage
    from evolucio.analysis.windows import within
    from evolucio.experiments.design import PairPolicy, load_design
    from evolucio.persistence.local import DEFAULT_RUNS_ROOT

    if execution_dir is not None and (registered or runs_root is not None):
        raise typer.BadParameter("--execution-dir is exclusive with registered or standalone runs")
    if environment_control is not None and temporal_metric is None:
        raise typer.BadParameter("--environment-control requires --temporal-metric")
    design = load_design(path)
    default_execution = Path("experiments") / str(design.experiment_id) / str(design.design_version)
    choices: dict[tuple[str, int], str] = {}
    for value in select_run or []:
        key, separator, run_id = value.partition("=")
        case, seed_separator, seed = key.rpartition(":")
        if not separator or not seed_separator or not run_id:
            raise typer.BadParameter("--select-run requires CASE:SEED=RUN_ID")
        cell = (case, int(seed))
        if cell in choices:
            raise typer.BadParameter("duplicate explicit selection")
        choices[cell] = run_id

    policy = PairPolicy.AVAILABLE_PAIRS if allow_available_pairs else None
    bootstrap = None if bootstrap_seed is None else BootstrapPolicy(bootstrap_seed)
    if registered:
        from .database_commands import open_registry

        registry, engine = open_registry()
        try:
            result = compare_experiment(
                design,
                run_provider=RegisteredRunProvider(registry, runs_root or DEFAULT_RUNS_ROOT),
                comparison_policy=policy,
                choices=choices,
                bootstrap=bootstrap,
            )
        finally:
            engine.dispose()
    else:
        provider = (
            LocalRunProvider(runs_root)
            if runs_root is not None
            else LocalRunProvider(execution_dir or default_execution, planned_execution=True)
        )
        result = compare_experiment(
            design,
            run_provider=provider,
            comparison_policy=policy,
            choices=choices,
            bootstrap=bootstrap,
        )
    temporal_tables = {}
    temporal_context: dict[str, object] = {}
    if temporal_metric is not None:
        cells = {
            (case, seed)
            for report in result.comparability_reports
            if report.comparison_eligible
            for case in (report.variant_case_id, design.baseline_case_id)
            for seed in report.paired_seeds
        }
        runs = {
            cell: run for cell, run in result.run_selection_report.selected.items() if cell in cells
        }
        if runs:
            window = AnalysisWindow(0, design.total_steps)
            alignment = AlignmentPolicy.COMMON_STEPS if common_steps else AlignmentPolicy.EXACT
            series = aligned_series(runs, temporal_metric, window, alignment)
            temporal_tables = {
                "time_series_summary": summarize_time_series(series),
                "paired_time_series": paired_time_series(series, design.baseline_case_id),
            }
            temporal_context.update(
                metric_name=temporal_metric,
                window=asdict(window),
                alignment=alignment.value,
                observed_steps=series.select("step")
                .unique()
                .sort("step")
                .collect()
                .get_column("step")
                .to_list(),
            )
            if environment_control is not None:
                contrast = next(
                    (
                        report
                        for report in result.comparability_reports
                        if report.variant_case_id == environment_control
                        and report.comparison_eligible
                    ),
                    None,
                )
                if contrast is None:
                    raise typer.BadParameter("environment control has no eligible comparison")
                windows = environment_windows(result.experiment_design, environment_control)
                phases = {
                    name: series.filter(
                        within("step", Coverage(phase.start_step, phase.end_step_exclusive))
                    )
                    for name, phase in windows.items()
                }
                if any(
                    frame.select("step").limit(1).collect().is_empty() for frame in phases.values()
                ):
                    raise ValueError("no common observations in one of the environmental windows")
                temporal_tables["differential_change"] = differential_change(
                    phases["before"], phases["after"], design.baseline_case_id, environment_control
                )
                temporal_context.update(
                    environment_control=environment_control,
                    environment_windows={key: asdict(phase) for key, phase in windows.items()},
                )
    target = output_dir or Path("comparisons") / result.comparison_id
    export_comparison(
        result, target, temporal_tables=temporal_tables, temporal_context=temporal_context or None
    )
    typer.echo(render_report(result))
    typer.echo(f"Informe desat a {target}")
    if not all(item.comparison_eligible for item in result.comparability_reports):
        raise typer.Exit(1)
