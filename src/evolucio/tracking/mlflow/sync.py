"""Reconstruct a terminal MLflow run from verified canonical data."""

# MLflow 3.x leaves some SDK attributes and parameters unannotated.
# pyright: reportUnknownMemberType=false

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from uuid import UUID

from mlflow import MlflowClient
from mlflow.entities import Metric, Run
from pydantic import BaseModel, ConfigDict

from evolucio.config import ExperimentConfig
from evolucio.config.freeze import canonical_json_and_hash
from evolucio.persistence.local import RunManifest, RunStatus
from evolucio.persistence.local.artifacts import (
    artifact_path,
    atomic_write_text,
    json_text,
    sha256_file,
)
from evolucio.persistence.parquet.index import INDEX_PATH as PARQUET_INDEX_PATH
from evolucio.persistence.postgres.models import MlflowRunBindingModel, RunModel
from evolucio.persistence.postgres.repository import PostgresRegistry

from .artifacts import artifact_sources, existing_artifacts, upload_artifact
from .client import create_tracking_client
from .errors import MlflowInputError, MlflowSyncError, MlflowTrackingIntegrityError
from .experiments import get_or_create_mlflow_experiment
from .metrics import FINAL_METRICS, STEP_METRICS, final_metrics, log_metrics, read_temporal_metrics
from .params import build_mlflow_params
from .runs import (
    build_tags,
    get_bound_run,
    get_or_create_mlflow_run,
    load_tracking_manifest,
    load_tracking_state,
    log_inputs,
    save_tracking_state,
    verify_tracking_identity,
)
from .types import MlflowTrackingPolicy, SyncStatus, TrackingState

TRACKING_TARGET = "metadata/tracking.json"
SUMMARY_FIELDS = (
    "alive_count",
    "births_total",
    "deaths_total",
    "active_lineage_count",
    "final_total_energy",
    "final_total_resources",
)


class CanonicalSummary(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    run_id: str
    final_step: int


def load_canonical_config(
    run_dir: Path, manifest: RunManifest, registry: PostgresRegistry
) -> tuple[RunModel, ExperimentConfig]:
    run_id = UUID(manifest.run_id)
    run = registry.get_run(run_id)
    snapshot = registry.get_config_snapshot(run_id)
    if run is None or run.experiment_id is None or snapshot is None:
        raise MlflowInputError("registered run, experiment, and config are required")
    fields = (
        "config_hash",
        "seed",
        "mode",
        "case_id",
        "batch_ordinal",
        "total_steps",
        "chunk_steps",
        "initial_step",
        "final_step",
        "status",
    )
    if any(getattr(run, name) != getattr(manifest, name) for name in fields):
        raise MlflowTrackingIntegrityError("manifest and PostgreSQL run metadata differ")
    path = artifact_path(run_dir, manifest.config_artifact)
    payload: object = json.loads(path.read_bytes())
    _, local_hash = canonical_json_and_hash(payload)
    _, snapshot_hash = canonical_json_and_hash(snapshot.config_json)
    entry = next(
        item for item in manifest.artifacts if item.relative_path == manifest.config_artifact
    )
    if (
        local_hash != manifest.config_hash
        or snapshot_hash != manifest.config_hash
        or snapshot.config_hash != manifest.config_hash
        or snapshot.artifact_sha256 != entry.sha256
        or snapshot.artifact_relative_path != manifest.config_artifact
        or sha256_file(path) != entry.sha256
    ):
        raise MlflowTrackingIntegrityError("frozen configuration and PostgreSQL digests differ")
    config = ExperimentConfig.model_validate(payload)
    if config.seed != manifest.seed:
        raise MlflowTrackingIntegrityError("frozen configuration seed differs")
    return run, config


def terminal_metrics(
    run_dir: Path, manifest: RunManifest, registry: PostgresRegistry
) -> Iterator[Metric]:
    if manifest.status != RunStatus.COMPLETED:
        return
    path = artifact_path(run_dir, "summary.json")
    entry = next(
        (item for item in manifest.artifacts if item.relative_path == "summary.json"), None
    )
    if entry is None or sha256_file(path) != entry.sha256:
        raise MlflowTrackingIntegrityError("summary digest differs from the manifest")
    summary = CanonicalSummary.model_validate_json(path.read_bytes())
    database = registry.get_run_summary(UUID(manifest.run_id))
    data = summary.model_dump()
    if (
        summary.run_id != manifest.run_id
        or summary.final_step != manifest.final_step
        or database is None
        or any(getattr(database, field) != data.get(field) for field in SUMMARY_FIELDS)
    ):
        raise MlflowTrackingIntegrityError("local and PostgreSQL summaries differ")
    for field in SUMMARY_FIELDS:
        data[field] = getattr(database, field)
    yield from final_metrics(data, summary.final_step, metric_timestamp(manifest))


def metric_timestamp(manifest: RunManifest) -> int:
    return int(datetime.fromisoformat(manifest.created_at_utc).timestamp() * 1000)


def desired_metrics(
    run_dir: Path, manifest: RunManifest, registry: PostgresRegistry, state: TrackingState
) -> Iterator[Metric]:
    yield from terminal_metrics(run_dir, manifest, registry)
    entries = {entry.relative_path: entry.sha256 for entry in manifest.artifacts}
    digest = entries.get(PARQUET_INDEX_PATH)
    if digest is None:
        if manifest.status is RunStatus.COMPLETED:
            raise MlflowTrackingIntegrityError("completed run lacks the Parquet index")
        return
    path = artifact_path(run_dir, PARQUET_INDEX_PATH)
    if not path.is_file() or sha256_file(path) != digest:
        raise MlflowTrackingIntegrityError("Parquet index digest differs from the manifest")
    yield from read_temporal_metrics(
        run_dir, manifest.run_id, state.metrics_interval, metric_timestamp(manifest)
    )


def metric_history(client: MlflowClient, run_id: str) -> dict[tuple[str, int], float]:
    history: dict[tuple[str, int], float] = {}
    for key in (*STEP_METRICS.values(), *FINAL_METRICS.values()):
        for point in client.get_metric_history(run_id, key):
            identity = (key, cast(int, point.step))
            prior = history.setdefault(identity, cast(float, point.value))
            if prior != point.value:
                raise MlflowTrackingIntegrityError(f"conflicting MLflow metric history: {key}")
    return history


def expected_status(manifest: RunManifest) -> str:
    if manifest.status is RunStatus.COMPLETED:
        return "FINISHED"
    if manifest.status is RunStatus.FAILED:
        return "FAILED"
    raise MlflowInputError("MLflow synchronization requires a terminal run")


def update_binding(binding: MlflowRunBindingModel, state: TrackingState) -> None:
    binding.sync_status = state.sync_status.value
    binding.last_error = state.last_error
    binding.updated_at = datetime.now(UTC)


def record_tracking_error(
    registry: PostgresRegistry,
    run_dir: Path,
    state: TrackingState,
    status: SyncStatus,
    message: str,
) -> TrackingState:
    failed = state.model_copy(
        update={
            "sync_status": status,
            "last_error": message,
            "final_sync_completed": False,
        }
    )
    with registry.lock_mlflow_run(UUID(state.evolucio_run_id)) as session:
        binding = session.get(MlflowRunBindingModel, UUID(state.evolucio_run_id))
        if binding is not None:
            verify_tracking_identity(
                state, state.evolucio_run_id, UUID(state.evolucio_experiment_id), binding
            )
            latest = load_tracking_state(run_dir)
            if binding.sync_status == SyncStatus.SYNCED and latest is not None:
                verify_tracking_identity(
                    latest, state.evolucio_run_id, UUID(state.evolucio_experiment_id), binding
                )
                if (
                    latest.final_sync_completed
                    and latest.sync_status == SyncStatus.SYNCED
                    and latest.mlflow_run_id == binding.mlflow_run_id
                ):
                    return latest
            failed = failed.model_copy(
                update={
                    "mlflow_run_id": binding.mlflow_run_id,
                    "mlflow_experiment_id": binding.mlflow_experiment_id,
                }
            )
            update_binding(binding, failed)
        save_tracking_state(run_dir, failed)
    return failed


def _missing_metrics(
    remote: Run,
    history: dict[tuple[str, int], float],
    run_dir: Path,
    manifest: RunManifest,
    registry: PostgresRegistry,
    state: TrackingState,
) -> Iterator[Metric]:
    for metric in desired_metrics(run_dir, manifest, registry, state):
        key = cast(str, metric.key)
        identity = (key, cast(int, metric.step))
        if identity not in history:
            if key.startswith(("final.", "runtime.")) and metric.key in remote.data.metrics:
                raise MlflowTrackingIntegrityError(
                    f"MLflow final metric step differs: {metric.key}"
                )
            yield metric
        elif history.pop(identity) != metric.value:
            raise MlflowTrackingIntegrityError(
                f"MLflow metric differs: {metric.key} step {metric.step}"
            )
    if history:
        raise MlflowTrackingIntegrityError("MLflow has unexpected metric steps")


def sync_run_to_mlflow(
    run_dir: Path,
    registry: PostgresRegistry,
    *,
    client: MlflowClient | None = None,
    tracking_uri: str | None = None,
    policy: MlflowTrackingPolicy | None = None,
) -> TrackingState:
    """Preflight immutable inputs and converge a run without re-executing it."""
    manifest = load_tracking_manifest(run_dir)
    status = expected_status(manifest)
    run, config = load_canonical_config(run_dir, manifest, registry)
    experiment_id = run.experiment_id
    assert experiment_id is not None
    stored = load_tracking_state(run_dir)
    requested = policy or MlflowTrackingPolicy()
    state = stored or TrackingState(
        evolucio_run_id=manifest.run_id,
        evolucio_experiment_id=str(experiment_id),
        metrics_interval=requested.metrics_interval,
        log_small_artifacts=requested.log_small_artifacts,
    )
    verify_tracking_identity(state, manifest.run_id, experiment_id)
    if policy is not None and (
        state.metrics_interval != policy.metrics_interval
        or state.log_small_artifacts != policy.log_small_artifacts
    ):
        raise MlflowTrackingIntegrityError("tracking policy differs from the recorded policy")

    try:
        remote_client = client or create_tracking_client(tracking_uri)
        experiment = get_or_create_mlflow_experiment(registry, remote_client, experiment_id)
        tags = build_tags(manifest, experiment_id)
        binding = get_or_create_mlflow_run(
            registry, remote_client, manifest, experiment.mlflow_experiment_id, tags
        )
        verify_tracking_identity(state, manifest.run_id, experiment_id, binding)
        state = state.model_copy(
            update={
                "mlflow_run_id": binding.mlflow_run_id,
                "mlflow_experiment_id": binding.mlflow_experiment_id,
            }
        )
        with registry.lock_mlflow_run(run.run_id) as session:
            binding = session.get(MlflowRunBindingModel, run.run_id)
            assert binding is not None
            latest = load_tracking_state(run_dir)
            if latest is not None:
                verify_tracking_identity(latest, manifest.run_id, experiment_id, binding)
                if policy is not None and (
                    latest.metrics_interval != policy.metrics_interval
                    or latest.log_small_artifacts != policy.log_small_artifacts
                ):
                    raise MlflowTrackingIntegrityError(
                        "tracking policy differs from the recorded policy"
                    )
                state = latest.model_copy(
                    update={
                        "mlflow_run_id": binding.mlflow_run_id,
                        "mlflow_experiment_id": binding.mlflow_experiment_id,
                    }
                )
            remote = get_bound_run(remote_client, binding, tags)
            if remote.info.status not in ("RUNNING", status):
                raise MlflowTrackingIntegrityError("MLflow run termination status differs")
            sources = artifact_sources(run_dir, manifest) if state.log_small_artifacts else {}
            params = build_mlflow_params(config, manifest)
            # Validate the entire history before writing to MLflow. Re-read bounded
            # Parquet parts on the second pass to avoid retaining unsent telemetry.
            history = metric_history(remote_client, binding.mlflow_run_id)
            for _ in _missing_metrics(remote, history.copy(), run_dir, manifest, registry, state):
                pass
            log_inputs(remote_client, remote, params, tags)
            log_metrics(
                remote_client,
                binding.mlflow_run_id,
                _missing_metrics(remote, history, run_dir, manifest, registry, state),
            )
            present = existing_artifacts(
                remote_client,
                binding.mlflow_run_id,
                set(sources) | ({TRACKING_TARGET} if state.log_small_artifacts else set()),
            )
            hashes: dict[str, str] = {}
            for target, source in sources.items():
                digest = sha256_file(source)
                if target not in present or state.artifact_hashes.get(target) != digest:
                    upload_artifact(remote_client, binding.mlflow_run_id, target, source)
                hashes[target] = digest
            synced = state.model_copy(
                update={
                    "sync_status": SyncStatus.SYNCED,
                    "last_error": None,
                    "final_sync_completed": True,
                    "artifact_hashes": hashes,
                }
            )
            if remote.info.status != status:
                assert manifest.finished_at_utc is not None
                ended = int(datetime.fromisoformat(manifest.finished_at_utc).timestamp() * 1000)
                remote_client.set_terminated(binding.mlflow_run_id, status, end_time=ended)
            if state.log_small_artifacts and (latest != synced or TRACKING_TARGET not in present):
                with TemporaryDirectory() as directory:
                    projected = Path(directory) / "tracking.json"
                    atomic_write_text(projected, json_text(synced.model_dump(mode="json")))
                    upload_artifact(
                        remote_client, binding.mlflow_run_id, TRACKING_TARGET, projected
                    )
            if binding.sync_status != SyncStatus.SYNCED or binding.last_error is not None:
                update_binding(binding, synced)
            if latest != synced:
                save_tracking_state(run_dir, synced)
        return synced
    except Exception as exc:
        if isinstance(exc, MlflowTrackingIntegrityError):
            raise
        try:
            record_tracking_error(
                registry,
                run_dir,
                state,
                SyncStatus.FAILED,
                "MLflow terminal synchronization failed",
            )
        except Exception:
            exc.add_note("Could not persist the MLflow tracking failure")
        raise MlflowSyncError("MLflow terminal synchronization failed") from exc
