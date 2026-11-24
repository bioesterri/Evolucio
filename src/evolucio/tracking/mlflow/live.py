"""Live tracking hooks for already recorded simulations."""

from uuid import UUID

from mlflow import MlflowClient

from evolucio.host import ChunkTelemetryDTO
from evolucio.persistence.local import LocalRunRecorder
from evolucio.persistence.postgres.repository import PostgresRegistry

from .client import create_tracking_client
from .errors import MlflowSyncError
from .experiments import get_or_create_mlflow_experiment
from .metrics import MlflowChunkConsumer
from .params import build_mlflow_params
from .runs import (
    build_tags,
    get_bound_run,
    get_or_create_mlflow_run,
    log_inputs,
    save_tracking_state,
)
from .sync import load_canonical_config, metric_timestamp, record_tracking_error, sync_run_to_mlflow
from .types import MlflowTrackingPolicy, SyncStatus, TrackingState


class MlflowRunTracker:
    def __init__(
        self,
        registry: PostgresRegistry,
        experiment_id: UUID,
        policy: MlflowTrackingPolicy,
        *,
        client: MlflowClient | None = None,
    ) -> None:
        self.registry = registry
        self.experiment_id = experiment_id
        self.policy = policy
        self.client = client
        self.state: TrackingState | None = None
        self.consumer: MlflowChunkConsumer | None = None
        self.failures: dict[str, str] = {}

    def created(self, recorder: LocalRunRecorder) -> None:
        self.consumer = None
        self.state = TrackingState(
            evolucio_run_id=recorder.run_id,
            evolucio_experiment_id=str(self.experiment_id),
            metrics_interval=self.policy.metrics_interval,
            log_small_artifacts=self.policy.log_small_artifacts,
        )
        try:
            manifest = recorder.manifest
            _, config = load_canonical_config(recorder.run_dir, manifest, self.registry)
            client = self.client or create_tracking_client()
            experiment = get_or_create_mlflow_experiment(self.registry, client, self.experiment_id)
            tags = build_tags(manifest, self.experiment_id, recorder.run_dir)
            binding = get_or_create_mlflow_run(
                self.registry, client, manifest, experiment.mlflow_experiment_id, tags
            )
            self.state = self.state.model_copy(
                update={
                    "mlflow_run_id": binding.mlflow_run_id,
                    "mlflow_experiment_id": binding.mlflow_experiment_id,
                }
            )
            save_tracking_state(recorder.run_dir, self.state)
            log_inputs(
                client,
                get_bound_run(client, binding, tags),
                build_mlflow_params(config, manifest),
                tags,
            )
            self.consumer = MlflowChunkConsumer(
                client, binding.mlflow_run_id, self.policy, metric_timestamp(manifest)
            )
            self.client = client
        except Exception as exc:
            message = "MLflow could not initialize run tracking"
            self._error(recorder, SyncStatus.FAILED, message)
            if self.policy.required:
                raise MlflowSyncError(message) from exc

    def consume(self, recorder: LocalRunRecorder, telemetry: ChunkTelemetryDTO) -> None:
        if self.consumer is not None:
            try:
                self.consumer(telemetry)
            except Exception:
                self.consumer = None
                self._error(recorder, SyncStatus.DEGRADED, "MLflow live metrics logging failed")

    def finished(self, recorder: LocalRunRecorder) -> None:
        if self.state is None or self.state.evolucio_run_id != recorder.run_id:
            return
        if self.state.mlflow_run_id is None and self.policy.required:
            return
        try:
            self.state = sync_run_to_mlflow(
                recorder.run_dir, self.registry, client=self.client, policy=self.policy
            )
            self.failures.pop(recorder.run_id, None)
        except Exception:
            self._error(recorder, SyncStatus.FAILED, "MLflow terminal synchronization failed")

    def _error(self, recorder: LocalRunRecorder, status: SyncStatus, message: str) -> None:
        self.failures[recorder.run_id] = message
        assert self.state is not None
        try:
            self.state = record_tracking_error(
                self.registry, recorder.run_dir, self.state, status, message
            )
        except Exception:
            self.failures[recorder.run_id] = f"{message}. Could not persist tracking status"
