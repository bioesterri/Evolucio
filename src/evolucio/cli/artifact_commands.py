"""Read-only inspection and integrity checks for recorded local runs."""

from dataclasses import asdict
from pathlib import Path

import typer

from evolucio.persistence.local import load_run_manifest, manifest_to_payload, verify_local_run

from .formatting import json_text

artifacts_app = typer.Typer(help="Inspect and verify local run records without executing.")


def _inspect_or_verify(run_dir: Path, *, inspect: bool) -> None:
    try:
        manifest = load_run_manifest(run_dir / "manifest.json")
        verification = verify_local_run(run_dir)
    except (OSError, ValueError) as exc:
        typer.echo(f"Local run invalid: {exc}", err=True)
        raise typer.Exit(2) from exc
    if inspect:
        typer.echo(json_text(manifest_to_payload(manifest)))
    typer.echo(json_text(asdict(verification)))
    if not verification.ok:
        raise typer.Exit(1)


@artifacts_app.command("inspect")
def inspect(run_dir: Path) -> None:
    """Show recorded context and integrity without modifying any files."""
    _inspect_or_verify(run_dir, inspect=True)


@artifacts_app.command("verify")
def verify(run_dir: Path) -> None:
    """Return 0 for integrity, 1 for damage, and 2 for invalid input."""
    _inspect_or_verify(run_dir, inspect=False)
