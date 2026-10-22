"""Explicit, idempotent synchronization and read-only reconciliation."""

from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid5

from evolucio.persistence.local.artifacts import sha256_file
from evolucio.persistence.local.manifest import load_run_manifest
from evolucio.persistence.local.verification import verify_local_run
from evolucio.persistence.parquet.index import INDEX_PATH as PARQUET_INDEX_PATH
from evolucio.persistence.parquet.index import load_index as load_parquet_index
from evolucio.persistence.zarr.index import INDEX_PATH as ZARR_INDEX_PATH
from evolucio.persistence.zarr.index import load_index as load_zarr_index

from .errors import RegistryIntegrityError
from .models import DatasetIndexModel, StateCaptureModel
from .repository import PostgresRegistry
from .types import ReconciliationReport

_IDENTITY_NAMESPACE = UUID("a828559c-5709-4ecf-981b-84ec2853c0e4")


def sync_run_artifacts(run_dir: Path, registry: PostgresRegistry) -> None:
    """Index verified local metadata without modifying local bytes or replacing hashes."""
    verification = verify_local_run(run_dir)
    if not verification.ok:
        raise RegistryIntegrityError(f"local run integrity check failed: {verification}")
    manifest = load_run_manifest(run_dir / "manifest.json")
    run_id = UUID(manifest.run_id)
    run = registry.get_run(run_id)
    if run is None or run.config_hash != manifest.config_hash:
        raise RegistryIntegrityError("local run identity/config is absent or differs in PostgreSQL")
    for artifact in manifest.artifacts:
        registry.index_artifact(run_id, artifact)

    parquet_path = run_dir / PARQUET_INDEX_PATH
    if parquet_path.is_file():
        parquet = load_parquet_index(run_dir)
        index_hash = sha256_file(parquet_path)
        for kind, dataset in parquet.datasets.items():
            registry.index_dataset(
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
                    index_sha256=index_hash,
                    created_at=datetime.fromisoformat(
                        manifest.created_at_utc.replace("Z", "+00:00")
                    ),
                )
            )

    zarr_path = run_dir / ZARR_INDEX_PATH
    if zarr_path.is_file():
        zarr = load_zarr_index(run_dir)
        for capture in (*zarr.checkpoints, *zarr.snapshots):
            registry.index_state_capture(
                StateCaptureModel(
                    capture_id=uuid5(
                        _IDENTITY_NAMESPACE,
                        f"{run_id}:capture:{capture.kind}:{capture.state_step}",
                    ),
                    run_id=run_id,
                    capture_kind=capture.kind,
                    state_step=capture.state_step,
                    relative_path=capture.relative_path,
                    schema_name=f"simulation_state_{capture.kind}_v1",
                    schema_version=capture.schema_version,
                    byte_size=capture.byte_size,
                    sha256=capture.sha256,
                    created_at=datetime.fromisoformat(
                        capture.created_at_utc.replace("Z", "+00:00")
                    ),
                )
            )


def reconcile_run(run_dir: Path, registry: PostgresRegistry) -> ReconciliationReport:
    """Compare both indexes without repairing either side."""
    manifest = load_run_manifest(run_dir / "manifest.json")
    run_id = UUID(manifest.run_id)
    run = registry.get_run(run_id)
    if run is None:
        return ReconciliationReport(False, ("run",), (), (), ())
    local = {record.relative_path: record for record in manifest.artifacts}
    database = {record.relative_path: record for record in registry.list_artifacts(run_id)}
    missing_db = list(set(local) - set(database))
    missing_local = list(set(database) - set(local))
    hashes = list(
        sorted(
            path
            for path in set(local) & set(database)
            if local[path].sha256 != database[path].sha256
        )
    )
    parquet_path = run_dir / PARQUET_INDEX_PATH
    local_datasets: set[str] = (
        set(load_parquet_index(run_dir).datasets) if parquet_path.is_file() else set()
    )
    database_datasets = {record.dataset_kind for record in registry.list_datasets(run_id)}
    missing_db.extend(f"dataset:{kind}" for kind in local_datasets - database_datasets)
    missing_local.extend(f"dataset:{kind}" for kind in database_datasets - local_datasets)
    if parquet_path.is_file():
        parquet_hash = sha256_file(parquet_path)
        hashes.extend(
            f"dataset:{record.dataset_kind}"
            for record in registry.list_datasets(run_id)
            if record.dataset_kind in local_datasets and record.index_sha256 != parquet_hash
        )
    zarr_path = run_dir / ZARR_INDEX_PATH
    zarr = load_zarr_index(run_dir) if zarr_path.is_file() else None
    local_captures = (
        {
            (record.kind, record.state_step): record
            for record in (*zarr.checkpoints, *zarr.snapshots)
        }
        if zarr is not None
        else {}
    )
    database_captures = {
        (record.capture_kind, record.state_step): record
        for record in registry.list_state_captures(run_id)
    }
    missing_db.extend(
        f"capture:{kind}:{step}" for kind, step in set(local_captures) - set(database_captures)
    )
    missing_local.extend(
        f"capture:{kind}:{step}" for kind, step in set(database_captures) - set(local_captures)
    )
    hashes.extend(
        f"capture:{kind}:{step}"
        for (kind, step), local_record in local_captures.items()
        if (database_record := database_captures.get((kind, step))) is not None
        and database_record.sha256 != local_record.sha256
    )
    metadata: list[str] = []
    if run.config_hash != manifest.config_hash:
        metadata.append("config_hash")
    if run.status != manifest.status.value:
        metadata.append("status")
    return ReconciliationReport(
        not (missing_db or missing_local or hashes or metadata),
        tuple(sorted(missing_db)),
        tuple(sorted(missing_local)),
        tuple(sorted(hashes)),
        tuple(metadata),
    )
