"""Bulk index completed analysis within the caller's final transaction."""

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import polars as pl
from sqlalchemy import insert

from evolucio.analysis.summaries import AGENT_SCHEMA, AGENTS_PATH, LINEAGE_SCHEMA, LINEAGES_PATH
from evolucio.persistence.genomes import GenomeHighlightReason, read_genome_index
from evolucio.persistence.local.artifacts import artifact_path

from .errors import RegistryIntegrityError
from .models import (
    AgentSummaryModel,
    GenomeHighlightModel,
    GenomeRegistryModel,
    LineageSummaryModel,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


def insert_run_analysis(session: "Session", run_id: UUID, run_dir: Path) -> None:
    agents = pl.read_parquet(artifact_path(run_dir, AGENTS_PATH))
    lineages = pl.read_parquet(artifact_path(run_dir, LINEAGES_PATH))
    index = read_genome_index(run_dir)
    if (
        agents.schema != AGENT_SCHEMA
        or lineages.schema != LINEAGE_SCHEMA
        or index.run_id != str(run_id)
        or any(
            frame.filter(pl.col("run_id").ne_missing(str(run_id))).height
            for frame in (agents, lineages)
        )
    ):
        raise RegistryIntegrityError("invalid analysis schema or run ID")
    for frame, model in ((agents, AgentSummaryModel), (lineages, LineageSummaryModel)):
        for batch in frame.iter_slices(n_rows=2000):
            rows = batch.to_dicts()
            for row in rows:
                row["run_id"] = run_id
            session.execute(insert(model).execution_options(render_nulls=True), rows)

    selected = agents.filter(pl.col("agent_id").is_in([g.agent_id for g in index.genomes]))
    selected_agents = {row["agent_id"]: row for row in selected.iter_rows(named=True)}
    created = datetime.now(UTC)
    genomes: list[dict[str, object]] = []
    highlights: list[dict[str, object]] = []
    for record in index.genomes:
        agent = selected_agents.get(record.agent_id)
        if (
            agent is None
            or any(
                agent[field] != getattr(record, field)
                for field in ("genome_id", "lineage_id", "generation")
            )
            or (
                GenomeHighlightReason.FOUNDER in record.reasons
                and agent["parent_agent_id"] is not None
            )
            or (
                GenomeHighlightReason.FINAL_LINEAGE_REPRESENTATIVE in record.reasons
                and not agent["alive_at_end"]
            )
        ):
            raise RegistryIntegrityError(f"invalid highlighted agent {record.agent_id}")
        genome: dict[str, object] = record.model_dump(exclude={"reasons"})
        genome.update(run_id=run_id, created_at=created)
        genomes.append(genome)
        for reason, step in record.reasons.items():
            highlights.append(
                {
                    "run_id": run_id,
                    "genome_id": record.genome_id,
                    "reason": reason.value,
                    "selected_step": step,
                }
            )
    if genomes:
        session.execute(insert(GenomeRegistryModel), genomes)
        session.execute(insert(GenomeHighlightModel), highlights)
