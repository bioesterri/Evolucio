"""Typer commands and predictable process exit codes."""

import sys
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from evolucio import __version__
from evolucio.config.errors import ConfigError
from evolucio.runtime.modes import RunDiagnosticError, RunMode

from .artifact_commands import artifacts_app
from .batch_command import batch_app
from .capture_commands import checkpoint_app, snapshot_app
from .config_commands import inspect_config, validate_config
from .run_command import run_config

app = typer.Typer(
    help="Validate, inspect and execute Evolució configurations.",
    invoke_without_command=True,
    no_args_is_help=True,
)
app.add_typer(checkpoint_app, name="checkpoint")
app.add_typer(snapshot_app, name="snapshot")
config_app = typer.Typer(help="Validate and inspect normalized configurations.")
app.add_typer(config_app, name="config")
app.add_typer(batch_app, name="batch")
app.add_typer(artifacts_app, name="artifacts")


class OutputFormat(StrEnum):
    """Supported inspection representations."""

    TEXT = "text"
    JSON = "json"


@app.callback()
def root(version: Annotated[bool, typer.Option("--version", is_eager=True)] = False) -> None:
    """Expose the installed package version."""
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@config_app.command("validate")
def validate(path: Path) -> None:
    """Validate a configuration without executing."""
    typer.echo(validate_config(path))


@config_app.command("inspect")
def inspect(
    path: Path,
    format: Annotated[OutputFormat, typer.Option("--format")] = OutputFormat.TEXT,
) -> None:
    """Show normalized configuration and canonical identities."""
    typer.echo(inspect_config(path, format=format.value))


@app.command("run")
def run(
    path: Path,
    steps: Annotated[int, typer.Option("--steps", min=1)],
    chunk_steps: Annotated[int | None, typer.Option("--chunk-steps", min=1)] = None,
    mode: Annotated[RunMode, typer.Option("--mode")] = RunMode.EXPERIMENT,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    checkpoint_every: Annotated[int | None, typer.Option("--checkpoint-every", min=1)] = None,
    snapshot_every: Annotated[int | None, typer.Option("--snapshot-every", min=1)] = None,
    checkpoint_on_completion: Annotated[bool, typer.Option("--checkpoint-on-completion")] = False,
    snapshot_on_completion: Annotated[bool, typer.Option("--snapshot-on-completion")] = False,
    keep_last_checkpoints: Annotated[
        int | None, typer.Option("--keep-last-checkpoints", min=1)
    ] = None,
    output_root: Annotated[
        Path | None, typer.Option("--output-root", help="Local run root (default: runs). ")
    ] = None,
) -> None:
    """Execute an explicit horizon or show its plan with --dry-run."""
    from evolucio.persistence.zarr.schemas import ZarrCapturePolicy

    capture_policy = ZarrCapturePolicy(
        checkpoint_every,
        snapshot_every,
        checkpoint_on_completion,
        snapshot_on_completion,
        keep_last_checkpoints,
    )
    output = run_config(
        path,
        steps=steps,
        chunk_steps=chunk_steps,
        quiet=quiet,
        dry_run=dry_run,
        mode=mode,
        output_root=output_root,
        capture_policy=capture_policy,
    )
    if output:
        typer.echo(output)


def main(argv: Sequence[str] | None = None) -> int:
    """Run commands with results on stdout and errors on stderr."""
    try:
        app(args=None if argv is None else list(argv), prog_name="evolucio")
        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except (ConfigError, ValidationError) as exc:
        print(f"Configuration invalid: {exc}", file=sys.stderr)
        return 2
    except RunDiagnosticError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Execution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
