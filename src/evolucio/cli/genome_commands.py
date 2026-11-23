"""Inspect highlighted neural policies from local runs."""

from pathlib import Path

import typer

from .formatting import json_text

genome_app = typer.Typer(help="List and verify highlighted genomes.")


@genome_app.command("list")
def list_genomes(run_dir: Path) -> None:
    """Show selected genome identities and their reasons."""
    from evolucio.persistence.genomes import read_genome_index

    try:
        index = read_genome_index(run_dir)
    except (OSError, ValueError) as exc:
        typer.echo(f"Genome index invalid: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        json_text(
            [
                {
                    "genome_id": record.genome_id,
                    "reason": reason.value,
                    "lineage_id": record.lineage_id,
                    "generation": record.generation,
                    "hash_prefix": record.genome_hash[:12],
                }
                for record in index.genomes
                for reason in record.reasons
            ]
        )
    )


@genome_app.command("verify")
def verify_genomes(run_dir: Path) -> None:
    """Verify the byte and semantic identity of indexed genome artifacts."""
    from evolucio.persistence.genomes import verify_genome_store

    result = verify_genome_store(run_dir)
    typer.echo(json_text({"ok": result.ok, "errors": result.errors}))
    if not result.ok:
        raise typer.Exit(1)
