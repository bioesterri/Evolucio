"""Verify and materialize only the committed telemetry prefix."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import BadZipFile

import jax
import polars as pl

from evolucio.analysis.summaries import FOUNDER_SCHEMA, FOUNDERS_PATH
from evolucio.persistence.genomes import (
    GenomeHighlightReason,
    read_genome_index,
    read_genome_leaves,
)
from evolucio.persistence.local.artifacts import artifact_path, sha256_file
from evolucio.persistence.parquet.index import CommittedChunk, DatasetIndex
from evolucio.persistence.parquet.schemas import GENEALOGY_COLUMNS, SCHEMAS, STEP_COLUMNS

from .errors import HistoryIntegrityError

if TYPE_CHECKING:
    from evolucio.persistence.local.manifest import RunManifest
    from evolucio.persistence.parquet.writer import ParquetRunConsumer
    from evolucio.persistence.zarr.checkpoint import CheckpointRestore


def prefix_frames(
    source: Path, index: DatasetIndex, step: int
) -> Iterator[tuple[CommittedChunk, dict[str, pl.DataFrame]]]:
    if index.schema_version != 2:
        raise HistoryIntegrityError("legacy index lacks durable chunk coverage")
    parts = {
        name: {part.chunk_index: part for part in ds.parts} for name, ds in index.datasets.items()
    }
    expected_step = 0
    for ordinal, chunk in enumerate(index.chunks):
        if expected_step == step:
            break
        end = chunk.end_step_exclusive
        if (
            chunk.chunk_index != ordinal
            or chunk.start_step != expected_step
            or not expected_step < end <= step
            or set(chunk.counts) != set(SCHEMAS)
        ):
            raise HistoryIntegrityError("missing or invalid committed chunk interval")
        frames: dict[str, pl.DataFrame] = {}
        for name, schema in SCHEMAS.items():
            part = parts[name].get(ordinal)
            frame = pl.DataFrame(schema=schema)
            if part is not None:
                path = artifact_path(source, part.relative_path)
                if path.stat().st_size != part.byte_size or sha256_file(path) != part.sha256:
                    raise HistoryIntegrityError(f"part hash or size differs: {part.relative_path}")
                frame = pl.read_parquet(path)
                steps = frame[STEP_COLUMNS[name]]
                if frame.height != part.row_count or (steps.min(), steps.max()) != (
                    part.min_step,
                    part.max_step,
                ):
                    raise HistoryIntegrityError(f"part statistics differ: {part.relative_path}")
            if (
                frame.schema != schema
                or frame.height != chunk.counts[name]
                or frame.filter(
                    (pl.col("run_id") != index.run_id)
                    | pl.col("run_id").is_null()
                    | ~pl.col(STEP_COLUMNS[name]).is_between(expected_step, end, closed="left")
                ).height
            ):
                raise HistoryIntegrityError(f"invalid {name} data in chunk {ordinal}")
            frames[name] = frame
        interval = index.policy.metrics_interval
        if frames["metrics"]["step"].to_list() != list(
            range(-(-expected_step // interval) * interval, end, interval)
        ):
            raise HistoryIntegrityError("missing or duplicated metric sample")
        if not frames["births"].select(GENEALOGY_COLUMNS).equals(frames["genealogy"]):
            raise HistoryIntegrityError("births/genealogy projection mismatch")
        for event_kind, count_column in (("births", "birth_count"), ("deaths", "death_count")):
            actual = {
                event_step: count
                for event_step, count in frames[event_kind]
                .group_by(STEP_COLUMNS[event_kind])
                .len()
                .iter_rows()
            }
            if any(
                actual.get(metric_step, 0) != count
                for metric_step, count in frames["metrics"].select("step", count_column).iter_rows()
            ):
                raise HistoryIntegrityError(f"{event_kind} count disagrees with sampled metrics")
        yield chunk, frames
        expected_step = end
    if expected_step != step:
        raise HistoryIntegrityError("committed telemetry does not reach checkpoint")


def verify_history(
    source: Path, manifest: RunManifest, index: DatasetIndex, restored: CheckpointRestore
) -> dict[str, int]:
    founders_path = artifact_path(source, FOUNDERS_PATH)
    record = next((row for row in manifest.artifacts if row.relative_path == FOUNDERS_PATH), None)
    if record is None or sha256_file(founders_path) != record.sha256:
        raise HistoryIntegrityError("founders lack a matching registered artifact")
    founders = pl.read_parquet(founders_path)
    genomes = read_genome_index(source)
    if founders.schema != FOUNDER_SCHEMA or genomes.run_id != manifest.run_id:
        raise HistoryIntegrityError("invalid founder schema or genome index run_id")
    found_genomes: dict[int, tuple[int, int]] = {}
    for genome in genomes.genomes:
        if GenomeHighlightReason.FOUNDER in genome.reasons:
            try:
                read_genome_leaves(source, genome)
            except (OSError, ValueError, BadZipFile, EOFError) as exc:
                raise HistoryIntegrityError(f"invalid founder genome {genome.genome_id}") from exc
            found_genomes[genome.genome_id] = (genome.agent_id, genome.lineage_id)
    if found_genomes != {
        genome: (agent, lineage)
        for agent, genome, lineage in founders.select(
            "agent_id", "genome_id", "lineage_id"
        ).iter_rows()
    }:
        raise HistoryIntegrityError("founder genomes disagree with founder identities")

    identity = ("genome_id", "lineage_id", "generation", "birth_step")
    alive = {
        agent: (genome, lineage, generation, birth_step, -1)
        for agent, genome, lineage, generation, birth_step in founders.select(
            "agent_id", *identity
        ).iter_rows()
    }
    if len(alive) != founders.height:
        raise HistoryIntegrityError("duplicated founder identity")
    seen_agents = set(alive)
    seen_genomes = set(found_genomes)
    counts = dict.fromkeys(SCHEMAS, 0)
    for _, frames in prefix_frames(source, index, restored.state_step):
        for name, frame in frames.items():
            counts[name] += frame.height
        births = frames["births"].partition_by("birth_step", as_dict=True)
        deaths = frames["deaths"].partition_by("death_step", as_dict=True)
        for key in sorted(births.keys() | deaths.keys()):
            if key in deaths:
                for agent, genome, lineage, generation, parent in (
                    deaths[key]
                    .select("agent_id", "genome_id", "lineage_id", "generation", "parent_id")
                    .iter_rows()
                ):
                    origin = alive.pop(agent, None)
                    if (
                        origin is None
                        or origin[:3] != (genome, lineage, generation)
                        or origin[4] != parent
                    ):
                        raise HistoryIntegrityError("death does not match a living agent")
            if key in births:
                for birth in births[key].iter_rows(named=True):
                    agent, genome = birth["child_agent_id"], birth["child_genome_id"]
                    parent = alive.get(birth["parent_agent_id"])
                    if (
                        agent in seen_agents
                        or genome in seen_genomes
                        or parent is None
                        or parent[:3]
                        != (birth["parent_genome_id"], birth["lineage_id"], birth["generation"] - 1)
                        or (parent[3] >= birth["birth_step"] and parent[2] != 0)
                    ):
                        raise HistoryIntegrityError("invalid or duplicate birth identity")
                    alive[agent] = (
                        genome,
                        birth["lineage_id"],
                        birth["generation"],
                        birth["birth_step"],
                        birth["parent_agent_id"],
                    )
                    seen_agents.add(agent)
                    seen_genomes.add(genome)

    population = jax.device_get(restored.state.population)
    mask = population.alive
    if alive != dict(
        zip(
            population.agent_id[mask].tolist(),
            zip(
                *(getattr(population, name)[mask].tolist() for name in (*identity, "parent_id")),
                strict=True,
            ),
            strict=True,
        )
    ):
        raise HistoryIntegrityError("checkpoint population disagrees with the committed history")
    acc = jax.device_get(restored.accumulator)
    if not bool(acc.count_overflow) and (int(acc.births_total), int(acc.deaths_total)) != (
        counts["births"],
        counts["deaths"],
    ):
        raise HistoryIntegrityError("checkpoint event totals disagree with the committed history")
    ids = jax.device_get(restored.state.ids)
    if (
        seen_agents != set(range(int(ids.next_agent_id)))
        or seen_genomes != set(range(int(ids.next_genome_id)))
        or {lineage for _, lineage in found_genomes.values()}
        != set(range(int(ids.next_lineage_id)))
    ):
        raise HistoryIntegrityError("checkpoint ID counters disagree with the committed history")
    return counts


def import_history(
    source: Path, index: DatasetIndex, step: int, writer: ParquetRunConsumer
) -> None:
    for chunk, frames in prefix_frames(source, index, step):
        writer.commit_frames(
            chunk.chunk_index,
            chunk.start_step,
            chunk.end_step_exclusive,
            {
                name: frame.with_columns(pl.lit(writer.recorder.run_id).alias("run_id"))
                for name, frame in frames.items()
            },
        )
    writer.imported_chunks = writer.last_chunk + 1
