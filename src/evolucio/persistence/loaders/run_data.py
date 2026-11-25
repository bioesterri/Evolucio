"""Portable metadata and lazy access to run artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import UUID

import polars as pl

from evolucio.analysis.summaries import (
    AGENT_SUMMARY_SCHEMA_NAME,
    AGENTS_PATH,
    FOUNDERS_PATH,
    LINEAGE_SUMMARY_SCHEMA_NAME,
    LINEAGES_PATH,
)
from evolucio.config import ExperimentConfig
from evolucio.config.errors import ConfigError
from evolucio.host import SimulationSnapshotDTO
from evolucio.persistence.genomes import (
    INDEX_PATH as GENOME_PATH,
)
from evolucio.persistence.genomes import (
    GenomeIndex,
    GenomeLeaves,
    HighlightedGenomeRecord,
    read_genome_leaves,
)
from evolucio.persistence.local.artifacts import artifact_path
from evolucio.persistence.local.manifest import RunManifest, RunStatus
from evolucio.persistence.parquet.index import INDEX_PATH as PARQUET_PATH
from evolucio.persistence.parquet.index import DatasetIndex
from evolucio.persistence.validation.types import ValidationLevel
from evolucio.persistence.zarr.index import INDEX_PATH as ZARR_PATH
from evolucio.persistence.zarr.index import CheckpointRecord, SnapshotRecord, ZarrIndex
from evolucio.recovery.provenance import PROVENANCE_PATH, RecoveryProvenance, read_provenance

from .dispatch import load_versioned
from .errors import DataSchemaError, MissingArtifactError, RunDataCompatibilityError
from .registry import (
    CHECKPOINT,
    CONFIG,
    GENOMES,
    PARQUET,
    RECOVERY,
    SNAPSHOT,
    TRACKING,
    ZARR,
    SchemaDescriptor,
    get_schema_descriptor,
    identify,
    read_object,
)
from .tracking import TrackingState, load_tracking_state

if TYPE_CHECKING:
    from evolucio.persistence.postgres.repository import PostgresRegistry

ANALYSIS_FORMATS = {
    FOUNDERS_PATH: "founders_v1",
    AGENTS_PATH: AGENT_SUMMARY_SCHEMA_NAME,
    LINEAGES_PATH: LINEAGE_SUMMARY_SCHEMA_NAME,
}


@dataclass(frozen=True, slots=True)
class RunData:
    path: Path
    manifest: RunManifest
    config: ExperimentConfig
    summary: dict[str, object] | None
    schemas: tuple[SchemaDescriptor, ...]
    parquet: DatasetIndex | None
    zarr: ZarrIndex | None
    genomes: GenomeIndex | None
    recovery: RecoveryProvenance | None
    tracking: TrackingState | None = None
    tracking_metadata_error: str | None = None

    def artifact(self, relative: str) -> Path:
        if not any(record.relative_path == relative for record in self.manifest.artifacts):
            raise MissingArtifactError(relative)
        path = artifact_path(self.path, relative)
        if not path.is_file():
            raise MissingArtifactError(relative)
        return path

    def _scan_telemetry(self, name: str) -> pl.LazyFrame:
        self.artifact(PARQUET_PATH)
        assert self.parquet is not None
        schema = get_schema_descriptor(name, self.parquet.datasets[name].schema_version)
        return load_versioned(self.path, schema, pl.LazyFrame)

    def scan_metrics(self) -> pl.LazyFrame:
        return self._scan_telemetry("metrics")

    def scan_births(self) -> pl.LazyFrame:
        return self._scan_telemetry("births")

    def scan_deaths(self) -> pl.LazyFrame:
        return self._scan_telemetry("deaths")

    def scan_genealogy(self) -> pl.LazyFrame:
        return self._scan_telemetry("genealogy")

    def scan_analysis(self, relative: str) -> pl.LazyFrame:
        path = self.artifact(relative)
        metadata = pl.read_parquet_metadata(path)
        expected = ANALYSIS_FORMATS[relative]
        version = metadata.get("schema_version")
        if metadata.get("schema_name") != expected or version is None or not version.isdecimal():
            raise DataSchemaError(f"{relative}: missing or incompatible analysis schema")
        get_schema_descriptor(expected, int(version))
        return pl.scan_parquet(path)

    def scan_founders(self) -> pl.LazyFrame:
        return self.scan_analysis(FOUNDERS_PATH)

    def scan_agent_summaries(self) -> pl.LazyFrame:
        return self.scan_analysis(AGENTS_PATH)

    def scan_lineage_summaries(self) -> pl.LazyFrame:
        return self.scan_analysis(LINEAGES_PATH)

    def list_snapshots(self) -> tuple[SnapshotRecord, ...]:
        return tuple(self.zarr.snapshots) if self.zarr else ()

    def list_checkpoints(self) -> tuple[CheckpointRecord, ...]:
        return tuple(self.zarr.checkpoints) if self.zarr else ()

    def list_highlighted_genomes(self) -> tuple[HighlightedGenomeRecord, ...]:
        return self.genomes.genomes if self.genomes else ()

    def load_snapshot(self, step: int) -> SimulationSnapshotDTO:
        for record in self.list_snapshots():
            if record.state_step == step:
                schema = get_schema_descriptor(SNAPSHOT, record.schema_version)
                return load_versioned(
                    artifact_path(self.path, record.relative_path), schema, SimulationSnapshotDTO
                )
        raise MissingArtifactError(f"snapshot at state step {step}")

    def load_highlighted_genome(self, genome_id: int) -> GenomeLeaves:
        for record in self.list_highlighted_genomes():
            if record.genome_id == genome_id:
                return read_genome_leaves(self.path, record)
        raise MissingArtifactError(f"highlighted genome {genome_id}")


def read_run_metadata(run_dir: Path) -> RunData:
    path = run_dir.resolve()
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink():
        raise DataSchemaError("manifest.json cannot be a symlink")
    header = read_object(manifest_path)
    name = header.get("manifest_schema_name")
    if not isinstance(name, str):
        raise DataSchemaError("manifest_schema_name is required")
    descriptor = identify(header, name, version_key="manifest_schema_version")
    manifest = load_versioned(manifest_path, descriptor, RunManifest)
    if path.name != manifest.run_id:
        raise DataSchemaError("directory and manifest run_id differ")
    schemas = [descriptor]
    paths = {
        record.relative_path: artifact_path(path, record.relative_path)
        for record in manifest.artifacts
    }
    config_path = paths[manifest.config_artifact]
    config_payload = read_object(config_path)
    config_descriptor = identify(config_payload, CONFIG)
    schemas.append(config_descriptor)
    for name in ("observations", "policy", "genome"):
        section = config_payload.get(name)
        if not isinstance(section, dict):
            raise DataSchemaError(f"Frozen configuration requires {name}")
        schemas.append(identify(cast(dict[str, object], section), f"config.{name}"))
    try:
        config = load_versioned(config_path, config_descriptor, ExperimentConfig)
    except ConfigError as exc:
        raise DataSchemaError(str(exc)) from exc
    if config.seed != manifest.seed or config.schema_version != manifest.schemas.config:
        raise DataSchemaError("configuration and manifest metadata differ")
    for name, digest in (
        ("host_dto", manifest.schemas.host_dto),
        ("simulation_step", manifest.schemas.simulation_step),
        ("metric_definitions", manifest.schemas.metrics),
    ):
        if digest is not None:
            schemas.append(SchemaDescriptor(name, "digest", digest))
    summary = read_object(paths["summary.json"]) if "summary.json" in paths else None
    if summary is not None and any(
        type(summary.get(key)) is not type(value) or summary[key] != value
        for key, value in {
            "run_id": manifest.run_id,
            "seed": manifest.seed,
            "status": manifest.status.value,
            "steps_executed": manifest.steps_executed,
            "final_step": manifest.final_step,
        }.items()
    ):
        raise DataSchemaError("summary and manifest metadata differ")
    parquet = zarr = genomes = None
    if PARQUET_PATH in paths:
        schemas.append(identify(read_object(paths[PARQUET_PATH]), PARQUET))
        parquet = load_versioned(path, schemas[-1], DatasetIndex)
        schemas.extend(
            get_schema_descriptor(name, ds.schema_version) for name, ds in parquet.datasets.items()
        )
    if ZARR_PATH in paths:
        schemas.append(identify(read_object(paths[ZARR_PATH]), ZARR))
        zarr = load_versioned(path, schemas[-1], ZarrIndex)
        for capture_name, records in ((CHECKPOINT, zarr.checkpoints), (SNAPSHOT, zarr.snapshots)):
            if records:
                schemas.append(get_schema_descriptor(capture_name, records[0].schema_version))
        if (zarr.config_hash, zarr.compile_signature) != (
            manifest.config_hash,
            manifest.compile_signature,
        ):
            raise DataSchemaError("Zarr index disagrees with manifest context")
    if GENOME_PATH in paths:
        schemas.append(identify(read_object(paths[GENOME_PATH]), GENOMES))
        genomes = load_versioned(path, schemas[-1], GenomeIndex)
        for record in genomes.genomes:
            artifact_path(path, record.relative_path)
    if any(
        index.run_id != manifest.run_id for index in (parquet, zarr, genomes) if index is not None
    ):
        raise DataSchemaError("index and manifest run_id differ")
    if manifest.status is RunStatus.COMPLETED:
        if (parquet is not None and not parquet.complete) or (
            zarr is not None and not zarr.complete
        ):
            raise DataSchemaError("completed run has an incomplete index")
        if summary and "parquet_dataset_index" in summary and parquet is None:
            raise MissingArtifactError(PARQUET_PATH)
        if summary and "agents_total" in summary:
            for required in (FOUNDERS_PATH, AGENTS_PATH, LINEAGES_PATH, GENOME_PATH):
                if required not in paths:
                    raise MissingArtifactError(required)
    if PROVENANCE_PATH in paths:
        schemas.append(identify(read_object(paths[PROVENANCE_PATH]), RECOVERY))
    recovery = read_provenance(path, manifest)
    try:
        tracking = load_tracking_state(path)
        if tracking is not None:
            if tracking.evolucio_run_id != manifest.run_id:
                raise DataSchemaError("tracking metadata and manifest run_id differ")
            schemas.append(get_schema_descriptor(TRACKING, tracking.schema_version))
        tracking_error = None
    except (OSError, ValueError) as exc:
        tracking = None
        tracking_error = str(exc)
    return RunData(
        path,
        manifest,
        config,
        summary,
        tuple(schemas),
        parquet,
        zarr,
        genomes,
        recovery,
        tracking,
        tracking_error,
    )


def open_run(
    run_dir: Path,
    *,
    validation_level: ValidationLevel = ValidationLevel.METADATA,
    allow_partial: bool = False,
) -> RunData:
    from evolucio.persistence.validation.runner import evaluate_run

    data, report, error = evaluate_run(
        run_dir, validation_level=validation_level, allow_partial=allow_partial
    )
    if error is not None:
        if isinstance(error, (DataSchemaError, OSError)):
            raise error
        raise DataSchemaError(str(error)) from error
    if data is None or not report.validation_passed:
        raise RunDataCompatibilityError("; ".join(issue.message for issue in report.issues))
    return data


def open_registered_run(
    run_id: UUID,
    registry: "PostgresRegistry",
    storage_root: Path,
    *,
    validation_level: ValidationLevel = ValidationLevel.METADATA,
) -> RunData:
    record = registry.get_run(run_id)
    if record is None:
        raise MissingArtifactError(f"unregistered run {run_id}")
    return open_run(
        registered_run_path(run_id, record.local_run_path, storage_root),
        validation_level=validation_level,
    )


def registered_run_path(run_id: UUID, stored_path: str | None, storage_root: Path) -> Path:
    root = storage_root.resolve()
    default = artifact_path(root, str(run_id))
    if default.is_dir() or stored_path is None:
        return default
    registered = Path(stored_path)
    if not registered.is_absolute():
        registered = root / registered
    if registered.name != str(run_id) or not registered.is_relative_to(root):
        return default
    return artifact_path(root, registered.relative_to(root).as_posix())
