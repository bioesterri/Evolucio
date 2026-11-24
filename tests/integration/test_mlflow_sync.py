"""Repair, retries and conflict detection against durable tracking state."""

import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import UUID

import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from evolucio.persistence.postgres import PostgresRegistry
from evolucio.persistence.postgres.models import MlflowRunBindingModel
from evolucio.tracking.mlflow import reconcile_mlflow_run, sync_run_to_mlflow
from evolucio.tracking.mlflow.errors import MlflowSyncError, MlflowTrackingIntegrityError
from evolucio.tracking.mlflow.runs import load_tracking_state
from evolucio.tracking.mlflow.sync import TRACKING_TARGET
from evolucio.tracking.mlflow.types import SyncStatus
from tests.integration.postgres.conftest import database_url as database_url
from tests.integration.postgres.conftest import migrated_engine as migrated_engine
from tests.integration.test_mlflow_tracking import app, run_args, runner
from tests.mlflow_helpers import client as client
from tests.mlflow_helpers import create_experiment
from tests.mlflow_helpers import registry as registry
from tests.mlflow_helpers import tracking_uri as tracking_uri


def test_terminal_failure_repair_and_idempotence(
    tmp_path: Path,
    registry: PostgresRegistry,
    client: MlflowClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp_id, remote_exp = create_experiment(registry, client, tmp_path)
    with monkeypatch.context() as patch:

        def fail(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("forced artifact failure")

        patch.setattr(MlflowClient, "log_artifact", fail)
        attempt = runner.invoke(app, run_args(tmp_path, exp_id))
    assert attempt.exit_code == 1, (attempt.output, attempt.exception)
    run_dir = next((tmp_path / "runs").iterdir())
    run = registry.get_run(UUID(run_dir.name))
    binding = registry.get_mlflow_run_binding(UUID(run_dir.name))
    assert run is not None and run.status == "completed"
    assert binding is not None and binding.sync_status == "failed"
    failed_state = load_tracking_state(run_dir)
    assert failed_state is not None and failed_state.sync_status is SyncStatus.FAILED
    assert not reconcile_mlflow_run(run_dir, registry, client=client).ok

    repaired = runner.invoke(app, ["mlflow", "sync", str(run_dir)])
    assert repaired.exit_code == 0, repaired.output
    original = client.get_metric_history(binding.mlflow_run_id, "population.alive")
    with monkeypatch.context() as patch:
        patch.setattr(MlflowClient, "log_artifact", fail)
        state = sync_run_to_mlflow(run_dir, registry, client=client)
    assert state.sync_status is SyncStatus.SYNCED
    assert client.get_run(binding.mlflow_run_id).info.status == "FINISHED"
    assert len(client.search_runs([remote_exp])) == 1
    assert client.get_metric_history(binding.mlflow_run_id, "population.alive") == original
    assert reconcile_mlflow_run(run_dir, registry, client=client).ok
    path = client.download_artifacts(binding.mlflow_run_id, TRACKING_TARGET, str(tmp_path))
    assert Path(path).read_text() == (run_dir / "artifacts/mlflow/tracking.json").read_text()

    def unreadable(*_args: object, **_kwargs: object) -> None:
        raise MlflowException("backend error containing an infrastructure detail")

    with monkeypatch.context() as patch:
        patch.setattr(client, "get_metric_history", unreadable)
        with pytest.raises(
            MlflowSyncError, match="could not read the MLflow tracking projection"
        ) as error:
            reconcile_mlflow_run(run_dir, registry, client=client)
        assert isinstance(error.value.__cause__, MlflowException)
        assert "infrastructure detail" not in str(error.value)

    with registry.lock_mlflow_run(UUID(run_dir.name)) as session:
        existing = session.get(MlflowRunBindingModel, UUID(run_dir.name))
        assert existing is not None
        session.delete(existing)
    assert not reconcile_mlflow_run(run_dir, registry, client=client).ok
    recovered = sync_run_to_mlflow(run_dir, registry, client=client)
    assert recovered.mlflow_run_id == binding.mlflow_run_id
    assert len(client.search_runs([remote_exp])) == 1

    with closing(sqlite3.connect(tmp_path / "tracking.sqlite")) as connection, connection:
        connection.execute(
            "UPDATE params SET value = ? WHERE run_uuid = ? AND key = ?",
            ("999", binding.mlflow_run_id, "seed"),
        )
    assert reconcile_mlflow_run(run_dir, registry, client=client).param_mismatches == ("seed",)
    with pytest.raises(MlflowTrackingIntegrityError, match="params differ: seed"):
        sync_run_to_mlflow(run_dir, registry, client=client)
    with closing(sqlite3.connect(tmp_path / "tracking.sqlite")) as connection, connection:
        connection.execute(
            "UPDATE params SET value = ? WHERE run_uuid = ? AND key = ?",
            ("1729", binding.mlflow_run_id, "seed"),
        )

    client.log_metric(binding.mlflow_run_id, "population.alive", 999, step=0)
    assert not reconcile_mlflow_run(run_dir, registry, client=client).ok
    with pytest.raises(MlflowTrackingIntegrityError, match="metric"):
        sync_run_to_mlflow(run_dir, registry, client=client)
