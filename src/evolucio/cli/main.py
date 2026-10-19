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

from .config_commands import inspect_config, validate_config
from .run_command import run_config

app = typer.Typer(
    help="Validate, inspect and execute Evolució configurations.",
    invoke_without_command=True,
    no_args_is_help=True,
)
config_app = typer.Typer(help="Validate and inspect normalized configurations.")
app.add_typer(config_app, name="config")


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
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Execute an explicit horizon or show its plan with --dry-run."""
    output = run_config(path, steps=steps, chunk_steps=chunk_steps, quiet=quiet, dry_run=dry_run)
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
    except Exception as exc:
        print(f"Execution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
