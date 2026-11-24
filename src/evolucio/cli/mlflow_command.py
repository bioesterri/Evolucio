"""CLI for optional MLflow tracking infrastructure."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from .formatting import text_fields

if TYPE_CHECKING:
    from evolucio.persistence.postgres.recording import PostgresRunObserver
    from evolucio.tracking.mlflow.live import MlflowRunTracker

mlflow_app = typer.Typer(help="Check, synchronize, and verify derived experiment tracking.")
MlflowFlag = Annotated[bool, typer.Option("--mlflow/--no-mlflow")]
MlflowRequired = Annotated[bool, typer.Option("--mlflow-required/--mlflow-optional")]
MlflowInterval = Annotated[int, typer.Option("--mlflow-metrics-interval", min=1)]


def tracking_for(
    observer: PostgresRunObserver | None, enabled: bool, required: bool, interval: int
) -> MlflowRunTracker | None:
    if not enabled:
        return None
    if observer is None:
        raise typer.BadParameter("--mlflow requires --experiment-id")
    try:
        from evolucio.tracking.mlflow.live import MlflowRunTracker
        from evolucio.tracking.mlflow.types import MlflowTrackingPolicy
    except ImportError as exc:
        raise typer.BadParameter("MLflow requires the persistence extra") from exc
    return MlflowRunTracker(
        observer.registry, observer.experiment_id, MlflowTrackingPolicy(interval, required=required)
    )


def show_tracking_failures(
    tracking: MlflowRunTracker | None, *, fail_required: bool = True
) -> None:
    if tracking is None or not tracking.failures:
        return
    for run_id, message in tracking.failures.items():
        typer.echo(f"MLflow {run_id}: {message}", err=True)
    if fail_required and tracking.policy.required:
        raise typer.Exit(1)


@mlflow_app.command("status")
def tracking_status() -> None:
    try:
        from evolucio.tracking.mlflow import check_mlflow_connection
    except ImportError:
        configured = "yes" if os.getenv("MLFLOW_TRACKING_URI") else "no"
        typer.echo(f"tracking configured: {configured}\nconnection: error\nMLflow: unavailable")
        raise typer.Exit(1) from None
    status = check_mlflow_connection()
    configured = "yes" if status.configured else "no"
    connection = "ok" if status.connected else "error"
    typer.echo(
        f"tracking configured: {configured}\nconnection: {connection}\nMLflow: {status.version}"
    )
    if not status.connected:
        raise typer.Exit(1)


def _operate(run_dir: Path, *, verify: bool) -> None:
    try:
        from evolucio.persistence.postgres.registry import (
            create_registry_engine,
            create_session_factory,
        )
        from evolucio.persistence.postgres.repository import PostgresRegistry
        from evolucio.tracking.mlflow import reconcile_mlflow_run, sync_run_to_mlflow
        from evolucio.tracking.mlflow.errors import MlflowTrackingIntegrityError

        engine = create_registry_engine()
    except Exception:
        typer.echo("Could not configure the PostgreSQL and MLflow clients.", err=True)
        raise typer.Exit(2) from None
    try:
        registry = PostgresRegistry(create_session_factory(engine))
        if verify:
            result = reconcile_mlflow_run(run_dir, registry)
            typer.echo(text_fields(asdict(result)))
            if not result.ok:
                raise typer.Exit(1)
        else:
            state = sync_run_to_mlflow(run_dir, registry)
            typer.echo(
                text_fields(
                    {
                        "run_id": state.evolucio_run_id,
                        "mlflow_run_id": state.mlflow_run_id,
                        "sync_status": state.sync_status.value,
                    }
                )
            )
    except MlflowTrackingIntegrityError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    except typer.Exit:
        raise
    except Exception:
        typer.echo("Could not read the run or connect to the tracking services.", err=True)
        raise typer.Exit(2) from None
    finally:
        engine.dispose()


@mlflow_app.command("sync")
def sync(run_dir: Path) -> None:
    """Reconstruct tracking from a recorded run directory."""
    _operate(run_dir, verify=False)


@mlflow_app.command("verify")
def verify(run_dir: Path) -> None:
    """Compare a recorded run against MLflow without writes."""
    _operate(run_dir, verify=True)
