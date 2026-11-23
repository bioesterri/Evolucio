"""Explicit, idempotent synchronization and read-only reconciliation."""

import json
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid5

from evolucio.persistence.local.artifacts import artifact_path
from evolucio.persistence.local.manifest import RunManifest, load_run_manifest
from evolucio.persistence.local.verification import verify_local_run
from evolucio.persistence.parquet.index import INDEX_PATH as PARQUET_INDEX_PATH
from evolucio.persistence.parquet.index import load_index as load_parquet_index
from evolucio.persistence.zarr.index import INDEX_PATH as ZARR_INDEX_PATH
from evolucio.persistence.zarr.index import load_index as load_zarr_index
from evolucio.persistence.zarr.schemas import CHECKPOINT_SCHEMA_NAME, SNAPSHOT_SCHEMA_NAME

from .errors import RegistryIntegrityError
from .models import DatasetIndexModel, StateCaptureModel
from .repository import PostgresRegistry
from .types import ReconciliationReport, RunIndexes

_IDENTITY_NAMESPACE = UUID("a828559c-5709-4ecf-981b-84ec2853c0e4")


def verified_run_indexes(run_dir: Path) -> tuple[RunManifest, RunIndexes]:
    try:
        verification = verify_local_run(run_dir)
        if not verification.ok:
            raise RegistryIntegrityError(f"local run integrity check failed: {verification}")
        return _read_indexes(run_dir)
    except (OSError, ValueError) as exc:
        raise RegistryIntegrityError("could not verify local run indexes") from exc


def sync_run_artifacts(run_dir: Path, registry: PostgresRegistry) -> None:
    """Index verified local metadata atomically without changing local bytes."""
    manifest, indexes = verified_run_indexes(run_dir)
    registry.sync_indexes(UUID(manifest.run_id), indexes)


def _read_indexes(run_dir: Path) -> tuple[RunManifest, RunIndexes]:
    manifest = load_run_manifest(run_dir / "manifest.json")
    run_id = UUID(manifest.run_id)
    artifacts = {record.relative_path: record for record in manifest.artifacts}
    datasets: list[DatasetIndexModel] = []
    captures: list[StateCaptureModel] = []

    if (index := artifacts.get(PARQUET_INDEX_PATH)) is not None:
        if index.kind != "parquet_dataset_index":
            raise RegistryIntegrityError("invalid artifact kind for the Parquet index")
        parquet = load_parquet_index(run_dir)
        for kind, dataset in parquet.datasets.items():
            datasets.append(
                DatasetIndexModel(
                    dataset_id=uuid5(_IDENTITY_NAMESPACE, f"{run_id}:dataset:{kind}"),
                    run_id=run_id,
                    dataset_kind=kind,
                    storage_format="parquet",
                    relative_path=dataset.relative_directory,
                    schema_name=parquet.schema_name,
                    schema_version=dataset.schema_version,
                    row_count=dataset.row_count,
                    part_count=dataset.part_count,
                    min_step=dataset.min_step,
                    max_step=dataset.max_step,
                    complete=parquet.complete,
                    index_sha256=index.sha256,
                    created_at=datetime.fromisoformat(manifest.created_at_utc),
                )
            )

    if (index := artifacts.get(ZARR_INDEX_PATH)) is not None:
        if index.kind != "zarr_artifact_index":
            raise RegistryIntegrityError("invalid artifact kind for the Zarr index")
        zarr = load_zarr_index(run_dir)
        for capture in (*zarr.checkpoints, *zarr.snapshots):
            captures.append(
                StateCaptureModel(
                    capture_id=uuid5(
                        _IDENTITY_NAMESPACE,
                        f"{run_id}:capture:{capture.kind}:{capture.state_step}",
                    ),
                    run_id=run_id,
                    capture_kind=capture.kind,
                    state_step=capture.state_step,
                    relative_path=capture.relative_path,
                    schema_name=CHECKPOINT_SCHEMA_NAME
                    if capture.kind == "checkpoint"
                    else SNAPSHOT_SCHEMA_NAME,
                    schema_version=capture.schema_version,
                    byte_size=capture.byte_size,
                    sha256=capture.sha256,
                    created_at=datetime.fromisoformat(capture.created_at_utc),
                )
            )
    return manifest, RunIndexes(
        manifest.config_hash, manifest.artifacts, tuple(datasets), tuple(captures)
    )


def reconcile_run(run_dir: Path, registry: PostgresRegistry) -> ReconciliationReport:
    """Compare physical integrity and relational metadata without modifying either."""
    verification = verify_local_run(run_dir)
    manifest = load_run_manifest(run_dir / "manifest.json")
    run_id = UUID(manifest.run_id)
    run = registry.get_run(run_id)
    missing_db: set[str] = set()
    missing_local = set(verification.missing_artifacts)
    hashes = set(verification.hash_mismatches)
    metadata = {
        *verification.invalid_paths,
        *verification.parquet_errors,
        *verification.zarr_errors,
        *verification.genome_errors,
    }

    def compare(
        local: dict[str, object],
        database: dict[str, object],
        hash_field: str,
        fields: tuple[str, ...],
    ) -> None:
        missing_db.update(local.keys() - database.keys())
        missing_local.update(database.keys() - local.keys())
        for name in local.keys() & database.keys():
            if getattr(local[name], hash_field) != getattr(database[name], hash_field):
                hashes.add(name)
            if any(
                getattr(local[name], field) != getattr(database[name], field) for field in fields
            ):
                metadata.add(name)

    if run is None:
        missing_db.add("run")
    else:
        for field in (
            "status",
            "config_hash",
            "seed",
            "mode",
            "case_id",
            "batch_ordinal",
            "final_step",
        ):
            if getattr(run, field) != getattr(manifest, field):
                metadata.add(field)
        compare(
            {a.relative_path: a for a in manifest.artifacts},
            {a.relative_path: a for a in registry.list_artifacts(run_id)},
            "sha256",
            ("name", "kind", "media_type", "byte_size"),
        )
        if verification.ok:
            _, indexes = _read_indexes(run_dir)
            compare(
                {f"dataset:{d.dataset_kind}": d for d in indexes.datasets},
                {f"dataset:{d.dataset_kind}": d for d in registry.list_datasets(run_id)},
                "index_sha256",
                (
                    "relative_path",
                    "storage_format",
                    "schema_name",
                    "schema_version",
                    "row_count",
                    "part_count",
                    "min_step",
                    "max_step",
                    "complete",
                ),
            )
            compare(
                {f"capture:{c.capture_kind}:{c.state_step}": c for c in indexes.captures},
                {
                    f"capture:{c.capture_kind}:{c.state_step}": c
                    for c in registry.list_state_captures(run_id)
                },
                "sha256",
                ("relative_path", "schema_name", "schema_version", "byte_size"),
            )
        config = registry.get_config_snapshot(run_id)
        local_config = next(
            a for a in manifest.artifacts if a.relative_path == manifest.config_artifact
        )
        if config is None:
            missing_db.add("config_snapshot")
        else:
            if config.artifact_sha256 != local_config.sha256:
                hashes.add("config_snapshot")
            if (
                config.config_hash != manifest.config_hash
                or config.artifact_relative_path != manifest.config_artifact
                or config.schema_version != int(manifest.schemas.config.split(".")[0])
                or (
                    verification.ok
                    and config.config_json
                    != json.loads(
                        artifact_path(run_dir, manifest.config_artifact).read_text(encoding="utf-8")
                    )
                )
            ):
                metadata.add("config_snapshot")
        summary = registry.get_run_summary(run_id)
        if manifest.status.value == "completed" and summary is None:
            missing_db.add("run_summary")
        elif summary is not None:
            if manifest.status.value != "completed":
                missing_local.add("run_summary")
            elif verification.ok:
                payload = json.loads(
                    artifact_path(run_dir, "summary.json").read_text(encoding="utf-8")
                )
                if any(
                    getattr(summary, field) != payload.get(field)
                    for field in (
                        "alive_count",
                        "births_total",
                        "deaths_total",
                        "active_lineage_count",
                        "final_total_energy",
                        "final_total_resources",
                    )
                ) or summary.population_extinct != (payload.get("alive_count") == 0):
                    metadata.add("run_summary")
    return ReconciliationReport(
        not (missing_db or missing_local or hashes or metadata),
        tuple(sorted(missing_db)),
        tuple(sorted(missing_local)),
        tuple(sorted(hashes)),
        tuple(sorted(metadata)),
    )
