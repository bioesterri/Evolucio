"""Minimal PostgreSQL schema and experiment administration commands."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from .formatting import text_fields

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from evolucio.persistence.postgres import PostgresRegistry


database_app = typer.Typer(help="Inspect and migrate the optional PostgreSQL registry.")
experiment_app = typer.Typer(help="Create and list relational experiment groups.")


@database_app.command("status")
def status() -> None:
    from evolucio.persistence.postgres import (
        check_database_connection,
        create_registry_engine,
        database_status,
    )

    engine = create_registry_engine()
    try:
        check_database_connection(engine)
        result = database_status(engine)
        typer.echo(
            json.dumps(
                {
                    "connected": True,
                    "current_revision": result.current_revision,
                    "head_revision": result.head_revision,
                    "up_to_date": result.up_to_date,
                },
                sort_keys=True,
            )
        )
    finally:
        engine.dispose()


@database_app.command("upgrade")
def upgrade() -> None:
    from evolucio.persistence.postgres import upgrade_database

    upgrade_database()
    typer.echo("Database upgraded to Alembic head.")


def _registry() -> tuple["PostgresRegistry", "Engine"]:
    from evolucio.persistence.postgres import (
        PostgresRegistry,
        create_registry_engine,
        create_session_factory,
    )

    engine = create_registry_engine()
    return PostgresRegistry(create_session_factory(engine)), engine


@experiment_app.command("create")
def create_experiment(
    name: Annotated[str, typer.Option("--name")],
    objective: Annotated[str | None, typer.Option("--objective")] = None,
) -> None:
    registry, engine = _registry()
    try:
        record = registry.create_experiment(name=name, objective=objective)
        typer.echo(str(record.experiment_id))
    finally:
        engine.dispose()


@experiment_app.command("list")
def list_experiments() -> None:
    registry, engine = _registry()
    try:
        for record in registry.list_experiments():
            typer.echo(f"{record.experiment_id}\t{record.status}\t{record.name}")
    finally:
        engine.dispose()


@database_app.command("verify-run")
def verify_run(run_dir: Path) -> None:
    """Compare local artifact and lifecycle identities with PostgreSQL."""
    from dataclasses import asdict

    from evolucio.persistence.postgres.reconcile import reconcile_run

    registry, engine = _registry()
    try:
        report = reconcile_run(run_dir, registry)
        typer.echo(text_fields(asdict(report)))
        if not report.ok:
            raise typer.Exit(1)
    finally:
        engine.dispose()


@database_app.command("sync-run")
def sync_run(run_dir: Path) -> None:
    """Publish verified artifact indexes and a recorded stop to PostgreSQL."""
    from evolucio.persistence.postgres.reconcile import sync_run_artifacts

    registry, engine = _registry()
    try:
        sync_run_artifacts(run_dir, registry)
    finally:
        engine.dispose()
