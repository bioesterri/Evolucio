"""The CLI batch projects one run per case and seed in one experiment."""

from pathlib import Path

from mlflow import MlflowClient

from evolucio.persistence.postgres import PostgresRegistry
from tests.integration.postgres.conftest import database_url as database_url
from tests.integration.postgres.conftest import migrated_engine as migrated_engine
from tests.integration.test_mlflow_tracking import app, runner
from tests.mlflow_helpers import client as client
from tests.mlflow_helpers import create_experiment
from tests.mlflow_helpers import registry as registry
from tests.mlflow_helpers import tracking_uri as tracking_uri
from tests.zarr_helpers import checkpoint_host_config


def test_batch_search_and_no_parent_runs(
    tmp_path: Path,
    registry: PostgresRegistry,
    client: MlflowClient,
) -> None:
    exp_id, remote_exp = create_experiment(registry, client, tmp_path)
    path = tmp_path / "config.json"
    path.write_text(checkpoint_host_config().model_dump_json())
    result = runner.invoke(
        app,
        [
            "batch",
            "run",
            "--config",
            f"baseline={path}",
            "--config",
            f"variant={path}",
            "--seeds",
            "1,2",
            "--steps",
            "1",
            "--chunk-steps",
            "1",
            "--experiment-id",
            exp_id,
            "--mlflow",
            "--output-root",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    runs = client.search_runs([remote_exp])
    assert len(runs) == 4
    assert len({run.data.tags["evolucio.run_id"] for run in runs}) == 4
    assert all("mlflow.parentRunId" not in run.data.tags for run in runs)
    matches = client.search_runs(
        [remote_exp],
        f"tags.`evolucio.experiment_id` = '{exp_id}' AND "
        "tags.`evolucio.case_id` = 'variant' AND params.seed = '2'",
    )
    assert len(matches) == 1
    assert matches[0].data.tags["evolucio.batch_ordinal"] == "4"
