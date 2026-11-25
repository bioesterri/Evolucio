"""Live tracking lifecycle, scientific identity, and independent failures."""

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

import pytest
from mlflow import MlflowClient
from typer.testing import CliRunner

from evolucio.cli.main import app
from evolucio.persistence.local import RunStatus, load_run_manifest
from evolucio.persistence.postgres import PostgresRegistry
from evolucio.runtime import initialization
from evolucio.tracking.mlflow import get_or_create_mlflow_experiment, reconcile_mlflow_run
from evolucio.tracking.mlflow.runs import load_tracking_state
from evolucio.tracking.mlflow.types import SyncStatus
from tests.integration.postgres.conftest import database_url as database_url
from tests.integration.postgres.conftest import migrated_engine as migrated_engine
from tests.mlflow_helpers import client as client
from tests.mlflow_helpers import create_experiment
from tests.mlflow_helpers import registry as registry
from tests.mlflow_helpers import tracking_uri as tracking_uri
from tests.zarr_helpers import checkpoint_host_config

runner = CliRunner()


def run_args(tmp_path: Path, experiment_id: str, *extra: str) -> list[str]:
    config = tmp_path / "config.json"
    config.write_text(checkpoint_host_config().model_dump_json())
    return [
        "run",
        str(config),
        "--steps",
        "3",
        "--chunk-steps",
        "2",
        "--experiment-id",
        experiment_id,
        "--output-root",
        str(tmp_path / "runs"),
        "--mlflow",
        "--mlflow-metrics-interval",
        "2",
        *extra,
    ]


def test_live_tracking_and_artifacts(
    tmp_path: Path,
    registry: PostgresRegistry,
    client: MlflowClient,
    tracking_uri: str,
) -> None:
    exp_id, remote_exp = create_experiment(registry, client, tmp_path)
    first_binding = get_or_create_mlflow_experiment(registry, client, UUID(exp_id))
    assert first_binding.mlflow_experiment_id == remote_exp
    assert (
        get_or_create_mlflow_experiment(registry, client, UUID(exp_id)).mlflow_experiment_id
        == remote_exp
    )
    args = run_args(tmp_path, exp_id, "--checkpoint-on-completion")
    config = checkpoint_host_config()
    population = config.population.model_copy(update={"initial_agents": 2})
    Path(args[1]).write_text(config.model_copy(update={"population": population}).model_dump_json())
    result = runner.invoke(app, args)
    assert result.exit_code == 0, (result.output, result.exception)
    run_dir = next((tmp_path / "runs").iterdir())
    binding = registry.get_mlflow_run_binding(UUID(run_dir.name))
    assert binding is not None and binding.mlflow_run_id != run_dir.name
    remote = client.get_run(binding.mlflow_run_id)
    assert remote.info.status == "FINISHED"
    assert remote.data.tags["evolucio.run_id"] == run_dir.name
    assert remote.data.tags["evolucio.experiment_id"] == exp_id
    assert remote.data.tags["evolucio.mode"] == "experiment"
    assert (
        remote.data.tags["evolucio.config_hash"]
        == load_run_manifest(run_dir / "manifest.json").config_hash
    )
    assert remote.data.params["seed"] == "1729"
    assert remote.data.params["total_steps"] == "3"
    assert remote.data.params["world.width"] == "6"
    assert [
        point.step for point in client.get_metric_history(binding.mlflow_run_id, "population.alive")
    ] == [0, 2]
    assert client.get_metric_history(binding.mlflow_run_id, "final.alive_count")[0].step == 3
    assert {item.path for item in client.list_artifacts(binding.mlflow_run_id)} == {
        "metadata",
        "indexes",
        "genomes",
    }
    assert {item.path for item in client.list_artifacts(binding.mlflow_run_id, "metadata")} == {
        "metadata/config.json",
        "metadata/manifest.json",
        "metadata/summary.json",
        "metadata/tracking.json",
    }
    assert {item.path for item in client.list_artifacts(binding.mlflow_run_id, "indexes")} == {
        "indexes/parquet-datasets.json",
        "indexes/zarr-index.json",
        "indexes/genomes-index.json",
    }
    assert all(
        item.path.endswith(".npz") and not item.is_dir
        for item in client.list_artifacts(binding.mlflow_run_id, "genomes")
    )
    assert reconcile_mlflow_run(run_dir, registry, client=client).ok
    status = runner.invoke(app, ["mlflow", "status"])
    assert status.exit_code == 0 and "connection: ok" in status.output
    verify = runner.invoke(app, ["mlflow", "verify", str(run_dir)])
    assert verify.exit_code == 0
    assert tracking_uri not in status.output


def test_runtime_failure_remains_primary(
    tmp_path: Path,
    registry: PostgresRegistry,
    client: MlflowClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp_id, _ = create_experiment(registry, client, tmp_path)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise ValueError("forced simulation failure")

    monkeypatch.setattr(initialization, "initialize_simulation", fail)
    result = runner.invoke(app, run_args(tmp_path, exp_id))
    assert type(result.exception) is ValueError
    run_dir = next((tmp_path / "runs").iterdir())
    assert load_run_manifest(run_dir / "manifest.json").status is RunStatus.FAILED
    run = registry.get_run(UUID(run_dir.name))
    binding = registry.get_mlflow_run_binding(UUID(run_dir.name))
    assert run is not None and run.status == "failed"
    assert binding is not None and binding.sync_status == "synced"
    remote = client.get_run(binding.mlflow_run_id)
    assert remote.info.status == "FAILED"
    assert remote.data.tags["error.type"] == "ValueError"
    assert "metadata/config.json" in {
        item.path for item in client.list_artifacts(binding.mlflow_run_id, "metadata")
    }
    assert reconcile_mlflow_run(run_dir, registry, client=client).ok


@pytest.mark.parametrize("required", [True, False])
def test_unavailable_tracking_policy(
    tmp_path: Path,
    registry: PostgresRegistry,
    monkeypatch: pytest.MonkeyPatch,
    required: bool,
) -> None:
    exp_id = str(registry.create_experiment(name="unavailable").experiment_id)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    args = run_args(tmp_path, exp_id, *([] if required else ["--mlflow-optional"]))
    dry = runner.invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0 and "enabled" in dry.output
    assert not (tmp_path / "runs").exists()
    result = runner.invoke(app, args)
    run_dir = next((tmp_path / "runs").iterdir())
    run = registry.get_run(UUID(run_dir.name))
    state = load_tracking_state(run_dir)
    assert run is not None and state is not None
    assert state.sync_status is SyncStatus.FAILED and state.mlflow_run_id is None
    if required:
        assert result.exit_code != 0 and run.started_at is None
    else:
        assert result.exit_code == 0 and run.status == "completed"
        assert "MLflow terminal synchronization failed" in result.output


def test_concurrent_experiment_creation_is_stable(
    tmp_path: Path,
    registry: PostgresRegistry,
    client: MlflowClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = registry.create_experiment(name="First name")
    original = client.create_experiment

    def create(name: str, *, tags: Mapping[str, str]) -> str:
        return original(name, artifact_location=(tmp_path / "artifacts").as_uri(), tags=dict(tags))

    monkeypatch.setattr(client, "create_experiment", create)

    def bind() -> str:
        return get_or_create_mlflow_experiment(
            registry, client, experiment.experiment_id
        ).mlflow_experiment_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = (executor.submit(bind) for _ in range(2))
        assert first.result() == second.result()
    with registry.lock_mlflow_experiment(experiment.experiment_id) as (_, record):
        record.name = "Renamed"
    assert bind() == first.result()
    matches = client.search_experiments(
        filter_string=f"tags.`evolucio.experiment_id` = '{experiment.experiment_id}'"
    )
    assert len(matches) == 1
    assert matches[0].name.startswith("Evolucio/First-name")


def test_recovery_tracking_replays_canonical_prefix(
    tmp_path: Path,
    registry: PostgresRegistry,
    client: MlflowClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evolucio.persistence.local.version_info import CodeVersionInfo
    from evolucio.persistence.postgres.recording import PostgresRunObserver
    from evolucio.recovery.runner import resume
    from evolucio.runtime.recorded_run import RecordedRunResult, run_recorded_simulation
    from evolucio.runtime.stop_control import StopController
    from evolucio.runtime.types import HostChunkBatch, RunPlan
    from evolucio.tracking.mlflow.live import MlflowRunTracker
    from evolucio.tracking.mlflow.types import MlflowTrackingPolicy

    identity = CodeVersionInfo("0.1.0.dev0", "b" * 40, False)
    monkeypatch.setattr(
        "evolucio.persistence.local.version_info.code_version_info", lambda: identity
    )
    monkeypatch.setattr("evolucio.recovery.assessment.code_version_info", lambda: identity)
    experiment, remote_experiment = create_experiment(registry, client, tmp_path)
    experiment_id = UUID(experiment)
    controller = StopController()

    def request_stop(_batch: HostChunkBatch) -> None:
        controller.request_stop()

    original = run_recorded_simulation(
        config=checkpoint_host_config(),
        plan=RunPlan(5, 2),
        output_root=tmp_path / "runs",
        observer=PostgresRunObserver(registry, experiment_id),
        stop_controller=controller,
        consumer=request_stop,
        tracking=MlflowRunTracker(
            registry,
            experiment_id,
            MlflowTrackingPolicy(),
            client=client,
        ),
    )
    source = registry.get_mlflow_run_binding(UUID(original.manifest.run_id))
    assert source is not None
    assert client.get_run(source.mlflow_run_id).info.status == "KILLED"
    recovered = resume(
        original.run_dir,
        observer=PostgresRunObserver(registry, experiment_id),
        tracking=MlflowRunTracker(
            registry,
            experiment_id,
            MlflowTrackingPolicy(required=False),
            client=client,
        ),
    )
    assert isinstance(recovered, RecordedRunResult)
    assert recovered.manifest.status is RunStatus.COMPLETED
    target = registry.get_mlflow_run_binding(UUID(recovered.manifest.run_id))
    assert target is not None and target.mlflow_run_id != source.mlflow_run_id
    remote = client.get_run(target.mlflow_run_id)
    assert remote.info.status == "FINISHED"
    assert remote.info.experiment_id == remote_experiment
    assert remote.data.tags["evolucio.root_run_id"] == original.manifest.run_id
    assert remote.data.tags["evolucio.recovered"] == "true"
    assert sorted(
        point.step for point in client.get_metric_history(target.mlflow_run_id, "population.alive")
    ) == [0, 1, 2, 3, 4]
    assert client.get_run(source.mlflow_run_id).info.status == "KILLED"
    assert reconcile_mlflow_run(recovered.run_dir, registry, client=client).ok

    from evolucio.persistence.postgres.errors import RegistryIntegrityError
    from evolucio.tracking.mlflow.sync import sync_run_to_mlflow

    def unavailable(*_args: object, **_kwargs: object) -> MlflowClient:
        raise RuntimeError("MLflow temporarily unavailable")

    with monkeypatch.context() as outage:
        outage.setattr("evolucio.tracking.mlflow.live.create_tracking_client", unavailable)
        outage.setattr("evolucio.tracking.mlflow.sync.create_tracking_client", unavailable)
        tracker = MlflowRunTracker(registry, experiment_id, MlflowTrackingPolicy(required=False))
        second = resume(
            original.run_dir,
            observer=PostgresRunObserver(registry, experiment_id),
            tracking=tracker,
        )
    assert isinstance(second, RecordedRunResult) and second.manifest.status is RunStatus.COMPLETED
    assert tracker.state and tracker.state.sync_status is SyncStatus.FAILED
    assert (
        sync_run_to_mlflow(second.run_dir, registry, client=client).sync_status is SyncStatus.SYNCED
    )
    with pytest.raises(RegistryIntegrityError, match="select one completed continuation"):
        registry.list_runs_for_comparison(experiment_id)
    assert {
        row.run_id
        for row in registry.list_runs_for_comparison(
            experiment_id, selected_runs=frozenset({UUID(second.manifest.run_id)})
        )
        if row.status == "completed"
    } == {UUID(second.manifest.run_id)}
