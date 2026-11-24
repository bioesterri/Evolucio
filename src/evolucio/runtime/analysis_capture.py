"""Bridge recorded telemetry and selected device policies to final host artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import jax
import numpy as np
import polars as pl

from evolucio.analysis.summaries import (
    AGENT_SUMMARY_SCHEMA_NAME,
    AGENTS_PATH,
    ANALYSIS_SUMMARY_SCHEMA_VERSION,
    FOUNDER_SCHEMA,
    FOUNDERS_PATH,
    LINEAGE_SUMMARY_SCHEMA_NAME,
    LINEAGES_PATH,
    SURVIVOR_SCHEMA,
    build_agent_summaries,
    build_lineage_summaries,
    select_final_representatives,
)
from evolucio.core.ids import NULL_ID
from evolucio.core.policy.batch import policy_at
from evolucio.persistence.genomes import (
    INDEX_PATH,
    GenomeHighlightReason,
    GenomeIndex,
    GenomeLeaves,
    HighlightedGenomeRecord,
    read_genome_index,
    read_genome_leaves,
    store_genome,
    verify_genome_store,
)
from evolucio.persistence.local.artifacts import artifact_path, sha256_file
from evolucio.persistence.local.manifest import load_run_manifest
from evolucio.persistence.parquet.index import INDEX_PATH as TELEMETRY_INDEX_PATH
from evolucio.persistence.parquet.reader import scan_births, scan_deaths
from evolucio.persistence.parquet.verify import verify_parquet_datasets

if TYPE_CHECKING:
    from evolucio.core.state import SimulationState
    from evolucio.persistence.local import LocalRunRecorder

PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"


@dataclass
class AnalysisCapture:
    recorder: LocalRunRecorder
    founders: pl.DataFrame
    genomes: dict[int, HighlightedGenomeRecord]
    historical_births: pl.LazyFrame | None = None
    historical_deaths: pl.LazyFrame | None = None

    @classmethod
    def start(cls, recorder: LocalRunRecorder, state: SimulationState) -> AnalysisCapture:
        population = jax.device_get(state.population)
        slots = np.flatnonzero(population.alive)
        founders = pl.DataFrame(
            {
                "agent_id": population.agent_id[slots],
                "lineage_id": population.lineage_id[slots],
                "genome_id": population.genome_id[slots],
                "generation": population.generation[slots],
                "birth_step": population.birth_step[slots],
                "initial_energy": population.energy[slots],
                "position_x": population.position[slots, 0],
                "position_y": population.position[slots, 1],
            },
            schema=FOUNDER_SCHEMA,
        ).sort("agent_id")
        capture = cls(recorder, founders, {})
        capture.write_frame(founders, FOUNDERS_PATH, "founders_v1")
        for slot in slots.tolist():
            capture.add_genome(
                state,
                slot,
                int(population.agent_id[slot]),
                int(population.genome_id[slot]),
                int(population.lineage_id[slot]),
                0,
                None,
                0,
                GenomeHighlightReason.FOUNDER,
            )
        capture.write_index()
        return capture

    @classmethod
    def resume(
        cls,
        recorder: LocalRunRecorder,
        source: Path,
        source_run_id: str,
        initial_step: int,
    ) -> AnalysisCapture:
        manifest = load_run_manifest(source / "manifest.json")
        if (
            manifest.run_id != source_run_id
            or manifest.config_hash != recorder.manifest.config_hash
        ):
            raise ValueError("checkpoint history does not belong to the restored run")
        index = read_genome_index(source)
        if index.run_id != source_run_id:
            raise ValueError("checkpoint source has an inconsistent genome index identity")
        required_paths = {FOUNDERS_PATH, INDEX_PATH, TELEMETRY_INDEX_PATH}
        historical_paths = {
            f"artifacts/analysis/historical_{kind}.parquet" for kind in ("births", "deaths")
        }
        artifacts = {record.relative_path: record for record in manifest.artifacts}
        if not required_paths <= artifacts.keys():
            raise ValueError("checkpoint source is missing registered history")
        for relative in required_paths | (historical_paths & artifacts.keys()):
            path = artifact_path(source, relative)
            record = artifacts[relative]
            if path.stat().st_size != record.byte_size or sha256_file(path) != record.sha256:
                raise ValueError(f"checkpoint history artifact differs: {relative}")

        report = verify_parquet_datasets(source)
        if not report.ok:
            raise ValueError(f"checkpoint source telemetry integrity check failed: {report}")
        capture = cls.import_founders(recorder, source)
        for kind, scan, step in (
            ("births", scan_births, "birth_step"),
            ("deaths", scan_deaths, "death_step"),
        ):
            relative = f"artifacts/analysis/historical_{kind}.parquet"
            frames = [scan(source)]
            history = artifact_path(source, relative)
            if relative in artifacts:
                frames.insert(0, pl.scan_parquet(history))
            historical = pl.concat(frames).filter(pl.col(step) < initial_step)
            historical = historical.with_columns(pl.lit(recorder.run_id).alias("run_id"))
            capture.write_frame(historical.collect(), relative, f"historical_{kind}_v1")
            if kind == "births":
                capture.historical_births = pl.scan_parquet(recorder.run_dir / relative)
            else:
                capture.historical_deaths = pl.scan_parquet(recorder.run_dir / relative)
        return capture

    @classmethod
    def import_founders(cls, recorder: LocalRunRecorder, source: Path) -> AnalysisCapture:
        index = read_genome_index(source)
        founders = pl.read_parquet(artifact_path(source, FOUNDERS_PATH))
        capture = cls(recorder, founders, {})
        capture.write_frame(founders, FOUNDERS_PATH, "founders_v1")
        for record in index.genomes:
            if GenomeHighlightReason.FOUNDER not in record.reasons:
                continue
            digest, path, sha, size = store_genome(
                recorder.run_dir,
                read_genome_leaves(source, record),
            )
            capture.genomes[record.genome_id] = record.model_copy(
                update={
                    "genome_hash": digest,
                    "relative_path": path,
                    "artifact_sha256": sha,
                    "byte_size": size,
                    "reasons": {GenomeHighlightReason.FOUNDER: 0},
                }
            )
        capture.write_index()
        return capture

    def add_genome(
        self,
        state: SimulationState,
        slot: int,
        agent_id: int,
        genome_id: int,
        lineage_id: int,
        generation: int,
        parent_genome_id: int | None,
        step: int,
        reason: GenomeHighlightReason,
    ) -> None:
        existing = self.genomes.get(genome_id)
        leaves = cast(
            "GenomeLeaves",
            tuple(
                np.asarray(leaf)
                for leaf in jax.device_get(jax.tree.leaves(policy_at(state.genomes, slot)))
            ),
        )
        digest, path, sha, size = store_genome(self.recorder.run_dir, leaves)
        record = HighlightedGenomeRecord(
            genome_id=genome_id,
            genome_hash=digest,
            agent_id=agent_id,
            parent_genome_id=parent_genome_id,
            lineage_id=lineage_id,
            generation=generation,
            reasons=(existing.reasons if existing else {}) | {reason: step},
            relative_path=path,
            artifact_sha256=sha,
            byte_size=size,
        )
        if existing and existing != record.model_copy(update={"reasons": existing.reasons}):
            raise ValueError(f"conflicting genome identity: {genome_id}")
        self.genomes[genome_id] = record

    def write_index(self) -> None:
        # Keep the founder index available even if simulation subsequently fails.
        from evolucio.persistence.local.artifacts import atomic_write_text

        index = GenomeIndex(
            run_id=self.recorder.run_id,
            genomes=tuple(self.genomes[key] for key in sorted(self.genomes)),
        )
        path = artifact_path(self.recorder.run_dir, INDEX_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, index.model_dump_json(indent=2) + "\n")

    def write_frame(self, frame: pl.DataFrame, relative: str, schema: str) -> None:
        path = artifact_path(self.recorder.run_dir, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(
            path,
            metadata={
                "schema_name": schema,
                "schema_version": str(ANALYSIS_SUMMARY_SCHEMA_VERSION),
            },
        )
        self.recorder.register_artifact(
            name=schema,
            kind="analysis_parquet",
            relative_path=relative,
            media_type=PARQUET_MEDIA_TYPE,
        )

    def finalize(
        self,
        state: SimulationState,
        final_step: int,
        *,
        births_total: int,
        deaths_total: int,
    ) -> dict[str, object]:
        population = jax.device_get(state.population)
        slots = np.flatnonzero(population.alive)
        survivors = pl.DataFrame(
            {
                "agent_id": population.agent_id[slots],
                "parent_agent_id": population.parent_id[slots],
                "lineage_id": population.lineage_id[slots],
                "genome_id": population.genome_id[slots],
                "generation": population.generation[slots],
                "birth_step": population.birth_step[slots],
                "final_age": population.age[slots],
                "final_energy": population.energy[slots],
            },
            schema=SURVIVOR_SCHEMA,
        ).with_columns(
            pl.when(pl.col("parent_agent_id") != NULL_ID)
            .then("parent_agent_id")
            .alias("parent_agent_id"),
        )
        births = scan_births(self.recorder.run_dir)
        deaths = scan_deaths(self.recorder.run_dir)
        if self.historical_births is not None and self.historical_deaths is not None:
            births = pl.concat([self.historical_births, births])
            deaths = pl.concat([self.historical_deaths, deaths])
        agents = build_agent_summaries(
            self.recorder.run_id,
            pl.scan_parquet(self.recorder.run_dir / FOUNDERS_PATH),
            births,
            deaths,
            survivors.lazy(),
            final_step,
        )
        if (
            agents.height != self.founders.height + births_total
            or agents.height - survivors.height != deaths_total
        ):
            raise ValueError("recorded lifecycle totals differ from the final accumulator")
        lineages = build_lineage_summaries(agents, births, final_step)
        parents = births.select(
            pl.col("child_agent_id").alias("agent_id"),
            "parent_genome_id",
        )
        representatives = (
            select_final_representatives(
                survivors.with_columns(pl.Series("slot", slots, dtype=pl.Int64)),
            )
            .lazy()
            .join(parents, on="agent_id", how="left")
            .collect()
        )
        for row in representatives.iter_rows(named=True):
            self.add_genome(
                state,
                row["slot"],
                row["agent_id"],
                row["genome_id"],
                row["lineage_id"],
                row["generation"],
                row["parent_genome_id"],
                final_step,
                GenomeHighlightReason.FINAL_LINEAGE_REPRESENTATIVE,
            )
        self.write_index()
        report = verify_genome_store(self.recorder.run_dir)
        if not report.ok:
            raise ValueError(f"genome store integrity check failed: {report.errors}")
        self.write_frame(agents, AGENTS_PATH, AGENT_SUMMARY_SCHEMA_NAME)
        self.write_frame(lineages, LINEAGES_PATH, LINEAGE_SUMMARY_SCHEMA_NAME)
        self.register_index()
        return {
            "agents_total": agents.height,
            "lineages_total": lineages.height,
            "highlighted_genomes": len(self.genomes),
        }

    def register_index(self) -> None:
        self.recorder.register_artifact(
            name="highlighted_genome_index",
            kind="highlighted_genome_index",
            relative_path=INDEX_PATH,
            media_type="application/json",
        )
