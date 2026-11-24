"""Run identity, safe tracking state, and immutable MLflow inputs."""

# MLflow 3.x leaves some SDK attributes and parameters unannotated.
# pyright: reportUnknownMemberType=false

from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

from mlflow import MlflowClient
from mlflow.entities import Param, Run, RunTag, ViewType

from evolucio.persistence.local import RunManifest, load_run_manifest
from evolucio.persistence.local.artifacts import artifact_path, atomic_write_text, json_text
from evolucio.persistence.postgres.models import MlflowRunBindingModel
from evolucio.persistence.postgres.repository import PostgresRegistry

from .errors import MlflowRunBindingError, MlflowTrackingIntegrityError
from .types import SyncStatus, TrackingState

TRACKING_PATH = "artifacts/mlflow/tracking.json"
RUN_TAG = "evolucio.run_id"
IDENTITY_TAGS = (RUN_TAG, "evolucio.experiment_id", "evolucio.config_hash")


def build_tags(manifest: RunManifest, experiment_id: UUID) -> dict[str, str]:
    values = {
        RUN_TAG: manifest.run_id,
        "evolucio.experiment_id": str(experiment_id),
        "evolucio.config_hash": manifest.config_hash,
        "evolucio.mode": manifest.mode.value,
        "evolucio.case_id": manifest.case_id,
        "evolucio.batch_ordinal": manifest.batch_ordinal,
        "evolucio.git_commit": manifest.code_version.git_commit,
        "evolucio.package_version": manifest.code_version.package_version,
    }
    if manifest.error is not None:
        values["error.type"] = manifest.error.error_type
        values["error.step"] = manifest.error.step
    return {key: str(value) for key, value in values.items() if value is not None}


def get_bound_run(
    client: MlflowClient, binding: MlflowRunBindingModel, tags: dict[str, str]
) -> Run:
    remote = client.get_run(binding.mlflow_run_id)
    if (
        remote.info.experiment_id != binding.mlflow_experiment_id
        or remote.info.lifecycle_stage != "active"
        or any(
            cast(dict[str, str], remote.data.tags).get(key) != tags[key] for key in IDENTITY_TAGS
        )
    ):
        raise MlflowTrackingIntegrityError("bound MLflow run identity differs")
    return remote


def get_or_create_mlflow_run(
    registry: PostgresRegistry,
    client: MlflowClient,
    manifest: RunManifest,
    experiment_id: str,
    tags: dict[str, str],
) -> MlflowRunBindingModel:
    run_id = UUID(manifest.run_id)
    try:
        with registry.lock_mlflow_run(run_id) as session:
            binding = session.get(MlflowRunBindingModel, run_id)
            if binding is not None:
                if binding.mlflow_experiment_id != experiment_id:
                    raise MlflowTrackingIntegrityError("run and experiment bindings disagree")
                get_bound_run(client, binding, tags)
                return binding

            matches = client.search_runs(
                [experiment_id],
                f"tags.`{RUN_TAG}` = '{run_id}'",
                run_view_type=ViewType.ALL,
                max_results=2,
            )
            if len(matches) > 1 or matches.token:
                raise MlflowTrackingIntegrityError("multiple MLflow runs match the UUID")
            if matches:
                remote = matches[0]
                if remote.info.lifecycle_stage != "active" or any(
                    cast(dict[str, str], remote.data.tags).get(key) != tags[key]
                    for key in IDENTITY_TAGS
                ):
                    raise MlflowTrackingIntegrityError("matching MLflow run identity differs")
            else:
                case = f"{manifest.case_id[:80]} " if manifest.case_id else ""
                remote = client.create_run(
                    experiment_id,
                    start_time=int(
                        datetime.fromisoformat(manifest.created_at_utc).timestamp() * 1000
                    ),
                    tags=tags,
                    run_name=f"{case}seed={manifest.seed} {manifest.run_id[:8]}",
                )
            now = datetime.now(UTC)
            binding = MlflowRunBindingModel(
                run_id=run_id,
                mlflow_run_id=remote.info.run_id,
                mlflow_experiment_id=experiment_id,
                sync_status=SyncStatus.ACTIVE.value,
                created_at=now,
                updated_at=now,
            )
            session.add(binding)
            return binding
    except MlflowTrackingIntegrityError:
        raise
    except Exception as exc:
        raise MlflowRunBindingError("could not bind the MLflow run") from exc


def load_tracking_manifest(run_dir: Path) -> RunManifest:
    path = run_dir / "manifest.json"
    if path.is_symlink():
        raise MlflowTrackingIntegrityError("manifest.json cannot be a symlink")
    manifest = load_run_manifest(path)
    if run_dir.resolve().name != manifest.run_id:
        raise MlflowTrackingIntegrityError("run directory and manifest identities differ")
    return manifest


def load_tracking_state(run_dir: Path) -> TrackingState | None:
    path = artifact_path(run_dir, TRACKING_PATH)
    if not path.is_file():
        return None
    try:
        return TrackingState.model_validate_json(path.read_bytes())
    except ValueError as exc:
        raise MlflowTrackingIntegrityError("invalid local MLflow tracking state") from exc


def save_tracking_state(run_dir: Path, state: TrackingState) -> None:
    path = artifact_path(run_dir, TRACKING_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json_text(state.model_dump(mode="json")))


def verify_tracking_identity(
    state: TrackingState,
    run_id: str,
    experiment_id: UUID,
    binding: MlflowRunBindingModel | None = None,
) -> None:
    if state.evolucio_run_id != run_id or state.evolucio_experiment_id != str(experiment_id):
        raise MlflowTrackingIntegrityError("local tracking and canonical identities differ")
    if (
        binding is not None
        and state.mlflow_run_id is not None
        and (
            state.mlflow_run_id != binding.mlflow_run_id
            or state.mlflow_experiment_id != binding.mlflow_experiment_id
        )
    ):
        raise MlflowTrackingIntegrityError("local tracking and PostgreSQL bindings differ")


def log_inputs(
    client: MlflowClient, remote: Run, params: dict[str, str], tags: dict[str, str]
) -> None:
    existing = cast(dict[str, str], remote.data.params)
    conflicts = sorted(
        key for key in params.keys() & existing.keys() if params[key] != existing[key]
    )
    if conflicts:
        raise MlflowTrackingIntegrityError(f"MLflow params differ: {', '.join(conflicts)}")
    missing = [Param(key, value) for key, value in params.items() if key not in existing]
    updated = [
        RunTag(key, value)
        for key, value in tags.items()
        if cast(dict[str, str], remote.data.tags).get(key) != value
    ]
    if missing or updated:
        client.log_batch(
            cast(str, remote.info.run_id), params=missing, tags=updated, synchronous=True
        )
