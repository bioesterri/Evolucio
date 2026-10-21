"""Indexed capture listing and read-only checkpoint verification."""

from pathlib import Path

import typer

from evolucio.persistence.zarr.index import list_snapshots, load_index

from .formatting import json_text

checkpoint_app = typer.Typer(help="List and verify immutable local checkpoints.")
snapshot_app = typer.Typer(help="List analytical snapshots (not resumable).")


@checkpoint_app.command("list")
def checkpoint_list(run_dir: Path) -> None:
    index = load_index(run_dir)
    typer.echo(
        json_text(
            [
                record.model_dump() | {"hash_prefix": record.sha256[:12]}
                for record in index.checkpoints
            ]
        )
    )


@checkpoint_app.command("verify")
def checkpoint_verify(checkpoint_path: Path) -> None:
    from evolucio.persistence.zarr.checkpoint import load_checkpoint

    try:
        restored = load_checkpoint(checkpoint_path=checkpoint_path)
        typer.echo(
            json_text(
                {
                    "ok": True,
                    "state_step": restored.state_step,
                    "sha256": restored.checkpoint_sha256,
                    "warnings": restored.warnings,
                }
            )
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Checkpoint invalid: {exc}", err=True)
        raise typer.Exit(1) from exc


@snapshot_app.command("list")
def snapshot_list(run_dir: Path) -> None:
    typer.echo(
        json_text(
            [
                record.model_dump() | {"hash_prefix": record.sha256[:12]}
                for record in list_snapshots(run_dir)
            ]
        )
    )
