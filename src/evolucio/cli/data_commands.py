"""Inspect versioned run data and report validation results."""

import json
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from evolucio.persistence.loaders.registry import get_supported_schemas
from evolucio.persistence.validation.types import ValidationLevel, ValidationSeverity

data_app = typer.Typer(help="Inspect and validate persisted run data.")


class OutputFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


def _emit(payload: dict[str, object], output: OutputFormat) -> None:
    if output is OutputFormat.JSON:
        typer.echo(json.dumps(payload, sort_keys=True, allow_nan=False))
    else:
        for name, value in payload.items():
            if isinstance(value, dict):
                typer.echo(f"{name}")
                for key, item in value.items():  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                    typer.echo(f"  {key} = {item}")
            else:
                typer.echo(f"{name} = {value}")


@data_app.command()
def schemas(
    output: Annotated[OutputFormat, typer.Option("--format")] = OutputFormat.TEXT,
) -> None:
    descriptors = get_supported_schemas()
    if output is OutputFormat.JSON:
        _emit({"schemas": [asdict(item) for item in descriptors]}, output)
    else:
        for item in descriptors:
            typer.echo(f"{item.name}\t{item.version}")


@data_app.command()
def inspect(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[OutputFormat, typer.Option("--format")] = OutputFormat.TEXT,
) -> None:
    from evolucio.persistence.validation.runner import evaluate_run

    run, report, error = evaluate_run(run_dir)
    if run is None:
        typer.echo(str(error), err=True)
        raise typer.Exit(1)
    _emit(
        {
            "run_id": run.manifest.run_id,
            "status": run.manifest.status.value,
            "schemas": [asdict(item) for item in run.schemas],
            "snapshots": [record.state_step for record in run.list_snapshots()],
            "checkpoints": [record.state_step for record in run.list_checkpoints()],
            "warnings": [issue.message for issue in report.issues],
        },
        output,
    )


@data_app.command()
def validate(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    level: Annotated[ValidationLevel, typer.Option()] = ValidationLevel.METADATA,
    output: Annotated[OutputFormat, typer.Option("--format")] = OutputFormat.TEXT,
    component: Annotated[list[str] | None, typer.Option("--component")] = None,
    allow_partial: Annotated[bool, typer.Option("--allow-partial")] = False,
) -> None:
    from evolucio.persistence.validation.runner import COMPONENTS, validate_run

    if component and set(component).difference(COMPONENTS):
        raise typer.BadParameter("Unknown component", param_hint="--component")
    report = validate_run(
        run_dir,
        validation_level=level,
        components=component or COMPONENTS,
        allow_partial=allow_partial,
    )
    payload = asdict(report)
    payload["errors"] = sum(issue.severity is ValidationSeverity.ERROR for issue in report.issues)
    payload["warnings"] = sum(
        issue.severity is ValidationSeverity.WARNING for issue in report.issues
    )
    _emit(payload, output)
    if not report.validation_passed:
        raise typer.Exit(1)
