"""Inspect, reconcile and resume recorded experiment trajectories."""

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from evolucio.persistence.local import RunStatus

from .formatting import text_fields

recovery_app = typer.Typer(help="Inspect and recover interrupted experiments.")


@recovery_app.command("inspect")
def inspect(run_dir: Path) -> None:
    from evolucio.recovery.assessment import inspect_run

    report = inspect_run(run_dir)
    typer.echo(text_fields(asdict(report) | {"recoverable": report.recoverable}))


@recovery_app.command("resume")
def resume(
    run_dir: Path,
    checkpoint_step: Annotated[int | None, typer.Option(min=0)] = None,
    latest_recoverable: bool = False,
    dry_run: bool = False,
    chunk_steps: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    from evolucio.recovery.assessment import RecoveryAssessment
    from evolucio.recovery.errors import RecoveryPlanError
    from evolucio.recovery.plan import RecoveryPlan
    from evolucio.recovery.provenance import read_provenance
    from evolucio.recovery.registration import recovery_adapters
    from evolucio.recovery.runner import resume as resume_run

    if (checkpoint_step is None) == (not latest_recoverable):
        raise typer.BadParameter("choose --checkpoint-step or --latest-recoverable")

    def show(plan: RecoveryPlan, assessment: RecoveryAssessment) -> None:
        typer.echo(
            text_fields(
                asdict(plan)
                | {
                    "source_status": assessment.source_status,
                    "history_to_import": f"[0, {plan.checkpoint_step})",
                    "new_execution": f"[{plan.checkpoint_step}, {plan.target_step})",
                    "recoverable": True,
                    "integrity_errors": assessment.integrity_errors,
                    "warnings": assessment.warnings,
                }
            )
        )

    adapters = nullcontext((None, None)) if dry_run else recovery_adapters(run_dir)
    try:
        with adapters as (observer, tracking):
            result = resume_run(
                run_dir,
                checkpoint_step=checkpoint_step,
                chunk_steps=chunk_steps,
                dry_run=dry_run,
                show_plan=show,
                observer=observer,
                tracking=tracking,
            )
    except RecoveryPlanError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if isinstance(result, RecoveryPlan):
        return
    provenance = read_provenance(result.run_dir, result.manifest)
    assert provenance is not None
    typer.echo(
        text_fields(
            {
                "source_run_id": provenance.source_run_id,
                "new_run_id": result.manifest.run_id,
                "root_run_id": provenance.root_run_id,
                "checkpoint_step": provenance.source_checkpoint_step,
                "inherited_steps": provenance.inherited_steps,
                "new_steps_executed": result.summary.steps_executed,
                "final_state_step": result.summary.final_step,
                "status": result.manifest.status,
                "manifest_path": str(result.run_dir / "manifest.json"),
                "tracking_status": (
                    tracking.state.sync_status
                    if tracking is not None and tracking.state is not None
                    else None
                ),
            }
        )
    )
    if result.manifest.status is RunStatus.STOPPED:
        raise typer.Exit(130)


@recovery_app.command("reconcile")
def reconcile(run_dir: Path) -> None:
    from evolucio.recovery.runner import reconcile as reconcile_run

    report = reconcile_run(run_dir)
    typer.echo(text_fields(asdict(report) | {"recoverable": report.recoverable}))
