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
