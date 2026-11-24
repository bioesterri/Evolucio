"""Read-only comparison of canonical and projected tracking data."""

# MLflow 3.x leaves some SDK attributes and parameters unannotated.
# pyright: reportUnknownMemberType=false

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from uuid import UUID

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from evolucio.persistence.local.artifacts import sha256_file
from evolucio.persistence.postgres.repository import PostgresRegistry

from .artifacts import artifact_sources, existing_artifacts
from .client import create_tracking_client
from .errors import MlflowSyncError, MlflowTrackingIntegrityError
from .experiments import EXPERIMENT_TAG
from .params import build_mlflow_params
from .runs import (
    IDENTITY_TAGS,
    TRACKING_PATH,
    build_tags,
    load_tracking_manifest,
    load_tracking_state,
)
from .sync import (
    TRACKING_TARGET,
    desired_metrics,
    expected_status,
    load_canonical_config,
    metric_history,
)
from .types import MlflowReconciliationReport, SyncStatus


def _reconcile(
    run_dir: Path,
    registry: PostgresRegistry,
    *,
    client: MlflowClient | None = None,
    tracking_uri: str | None = None,
) -> MlflowReconciliationReport:
    manifest = load_tracking_manifest(run_dir)
    status = expected_status(manifest)
    run, config = load_canonical_config(run_dir, manifest, registry)
    assert run.experiment_id is not None
    exp_binding = registry.get_mlflow_experiment_binding(run.experiment_id)
    binding = registry.get_mlflow_run_binding(UUID(manifest.run_id))
    state = load_tracking_state(run_dir)
    missing = tuple(
        name
        for name, value in (("experiment", exp_binding), ("run", binding), ("tracking.json", state))
        if value is None
    )
    if exp_binding is None or binding is None or state is None:
        return MlflowReconciliationReport(False, missing_binding=missing)

    differences: set[str] = set()
    tag_mismatches: set[str] = set()
    metric_mismatches: set[str] = set()
    if (
        state.evolucio_run_id != manifest.run_id
        or state.evolucio_experiment_id != str(run.experiment_id)
        or state.mlflow_run_id != binding.mlflow_run_id
        or state.mlflow_experiment_id != binding.mlflow_experiment_id
        or binding.mlflow_experiment_id != exp_binding.mlflow_experiment_id
    ):
        differences.add("binding identity")
    status_mismatch = (
        state.sync_status != binding.sync_status
        or state.sync_status != SyncStatus.SYNCED
        or state.last_error != binding.last_error
        or not state.final_sync_completed
    )

    remote_client = client or create_tracking_client(tracking_uri)
    try:
        experiment = remote_client.get_experiment(exp_binding.mlflow_experiment_id)
        remote = remote_client.get_run(binding.mlflow_run_id)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            return MlflowReconciliationReport(False, mismatches=("MLflow resource missing",))
        raise
    if (
        experiment.tags.get(EXPERIMENT_TAG) != str(run.experiment_id)
        or experiment.lifecycle_stage != "active"
    ):
        differences.add("experiment identity")
    if (
        remote.info.experiment_id != exp_binding.mlflow_experiment_id
        or remote.info.lifecycle_stage != "active"
    ):
        differences.add("run identity")
    status_mismatch |= cast(str, remote.info.status) != status
    tags = build_tags(manifest, run.experiment_id)
    for key, value in tags.items():
        if cast(dict[str, str], remote.data.tags).get(key) != value:
            prefix = "identity tag" if key in IDENTITY_TAGS else "tag"
            tag_mismatches.add(f"{prefix}: {key}")

    params = build_mlflow_params(config, manifest)
    remote_params = cast(dict[str, str], remote.data.params)
    missing_params = tuple(sorted(params.keys() - remote_params.keys()))
    param_mismatches = tuple(
        sorted(
            key for key in params.keys() & remote_params.keys() if remote_params[key] != params[key]
        )
    )

    missing_metrics: set[str] = set()
    try:
        history = metric_history(remote_client, binding.mlflow_run_id)
    except MlflowTrackingIntegrityError as exc:
        metric_mismatches.add(str(exc))
    else:
        for metric in desired_metrics(run_dir, manifest, registry, state):
            identity = (cast(str, metric.key), cast(int, metric.step))
            label = f"{metric.key}@{metric.step}"
            if identity not in history:
                missing_metrics.add(label)
            elif history.pop(identity) != metric.value:
                metric_mismatches.add(label)
        metric_mismatches.update(f"unexpected: {key}@{step}" for key, step in history)

    sources = artifact_sources(run_dir, manifest) if state.log_small_artifacts else {}
    targets = set(sources)
    if state.log_small_artifacts:
        targets.add(TRACKING_TARGET)
    missing_artifacts = targets - existing_artifacts(remote_client, binding.mlflow_run_id, targets)
    digests = {target: sha256_file(source) for target, source in sources.items()}
    for target, digest in digests.items():
        if state.artifact_hashes.get(target) != digest:
            differences.add(f"artifact digest: {target}")
    if state.log_small_artifacts:
        digests[TRACKING_TARGET] = sha256_file(run_dir / TRACKING_PATH)
    with TemporaryDirectory() as directory:
        for target in sorted(targets - missing_artifacts):
            copy = remote_client.download_artifacts(binding.mlflow_run_id, target, directory)
            if sha256_file(Path(copy)) != digests[target]:
                differences.add(f"remote artifact digest: {target}")

    return MlflowReconciliationReport(
        not (
            missing_params
            or param_mismatches
            or tag_mismatches
            or missing_metrics
            or metric_mismatches
            or missing_artifacts
            or status_mismatch
            or differences
        ),
        missing_params=missing_params,
        param_mismatches=param_mismatches,
        tag_mismatches=tuple(sorted(tag_mismatches)),
        metric_mismatches=tuple(sorted(metric_mismatches)),
        status_mismatch=status_mismatch,
        missing_metrics=tuple(sorted(missing_metrics)),
        missing_artifacts=tuple(sorted(missing_artifacts)),
        mismatches=tuple(sorted(differences)),
    )


def reconcile_mlflow_run(
    run_dir: Path,
    registry: PostgresRegistry,
    *,
    client: MlflowClient | None = None,
    tracking_uri: str | None = None,
) -> MlflowReconciliationReport:
    """Compare a terminal run without modifying either store."""
    try:
        return _reconcile(run_dir, registry, client=client, tracking_uri=tracking_uri)
    except MlflowException as exc:
        raise MlflowSyncError("could not read the MLflow tracking projection") from exc
