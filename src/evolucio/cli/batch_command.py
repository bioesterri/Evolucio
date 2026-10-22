"""Batch CLI: explicit matrix before execution and one completion line per run."""

from pathlib import Path
from typing import Annotated

import typer

from evolucio.experiments import (
    BatchDefinitionError,
    BatchExecutionError,
    BatchPlan,
    BatchPreflight,
    BatchRunResult,
    BatchRunSpec,
    BatchRunStatus,
    ExperimentCase,
    SeedSet,
    expand_batch_plan,
    preflight_batch,
    run_batch,
)
from evolucio.persistence.local import DEFAULT_RUNS_ROOT
from evolucio.runtime.modes import RunMode

batch_app = typer.Typer(help="Execute explicit configurations with paired master seeds.")


def _show_matrix(view: BatchPreflight, specs: tuple[BatchRunSpec, ...]) -> None:
    typer.echo(
        f"cases: {view.case_count}\nseeds: {view.seed_count}\nruns: {view.run_count}\n"
        f"mode: {view.mode.value}\nsteps/run: {view.total_steps_per_run}\n"
        f"chunk_steps: {view.chunk_steps}"
    )
    for case in view.cases:
        typer.echo(f"case: {case.case_id} path={case.path} config_hash={case.config_hash}")
    hashes = {case.case_id: dict(case.run_config_hashes) for case in view.cases}
    for spec in specs:
        typer.echo(
            f"{spec.ordinal:03d} {spec.case_id} seed={spec.seed} "
            f"config_hash={hashes[spec.case_id][spec.seed]}"
        )


def _show_summary(runs: tuple[BatchRunResult, ...], requested: int) -> None:
    completed = sum(run.status == BatchRunStatus.COMPLETED for run in runs)
    typer.echo(
        f"runs requested: {requested}\nruns executed: {len(runs)}\n"
        f"completed: {completed}\nfailed: {len(runs) - completed}"
    )
    for run in runs:
        if run.status == BatchRunStatus.FAILED:
            typer.echo(
                f"{run.case_id} seed={run.seed}: {run.error_type}: {run.error_message}", err=True
            )


@batch_app.command("run")
def batch_run(
    config: Annotated[list[str], typer.Option("--config", help="Repeat CASE_ID=PATH.")],
    seeds: Annotated[str, typer.Option("--seeds", help="Explicit comma-separated integers.")],
    steps: Annotated[int, typer.Option("--steps", min=1)],
    chunk_steps: Annotated[int, typer.Option("--chunk-steps", min=1)],
    mode: Annotated[RunMode, typer.Option("--mode")] = RunMode.EXPERIMENT,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    fail_fast: Annotated[bool, typer.Option("--fail-fast")] = False,
    output_root: Annotated[Path | None, typer.Option("--output-root")] = None,
) -> None:
    """Run the case by seed matrix sequentially, or inspect it without simulation."""
    try:
        cases: list[ExperimentCase] = []
        for item in config:
            case_id, separator, path = item.partition("=")
            if not separator or not path:
                raise ValueError("--config requires CASE_ID=PATH")
            cases.append(ExperimentCase(case_id, Path(path)))
        seed_set = SeedSet(tuple(int(value.strip()) for value in seeds.split(",")))
        plan = BatchPlan(tuple(cases), seed_set, steps, chunk_steps, mode, not fail_fast)
    except (ValueError, TypeError) as exc:
        raise BatchDefinitionError(str(exc)) from exc
    if dry_run:
        _show_matrix(preflight_batch(plan), expand_batch_plan(plan))
        return

    def progress(run: BatchRunResult) -> None:
        typer.echo(
            f"[{run.ordinal}/{plan.run_count}] {run.case_id} seed={run.seed} ... {run.status.value}"
            f" run_id={run.run_id} run_dir={run.run_dir}"
        )

    try:
        result = run_batch(
            plan=plan,
            on_preflight=_show_matrix,
            on_result=progress,
            output_root=DEFAULT_RUNS_ROOT if output_root is None else output_root,
        )
    except BatchExecutionError as exc:
        _show_summary(exc.runs, exc.requested_run_count)
        raise typer.Exit(1) from exc
    _show_summary(result.runs, result.requested_run_count)
    if result.failed_run_count:
        raise typer.Exit(1)
