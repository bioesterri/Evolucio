"""Test infrastructure for a real local MLflow SQL store and PostgreSQL registry."""

from pathlib import Path

import pytest
from mlflow import MlflowClient
from sqlalchemy import Engine

from evolucio.persistence.postgres import PostgresRegistry, create_session_factory


@pytest.fixture
def registry(migrated_engine: Engine) -> PostgresRegistry:
    return PostgresRegistry(create_session_factory(migrated_engine))


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    uri = f"sqlite:///{tmp_path / 'tracking.sqlite'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


@pytest.fixture
def client(tracking_uri: str) -> MlflowClient:
    return MlflowClient(tracking_uri=tracking_uri)


def create_experiment(
    registry: PostgresRegistry, client: MlflowClient, tmp_path: Path
) -> tuple[str, str]:
    experiment = registry.create_experiment(name="baseline and variant")
    canonical_id = str(experiment.experiment_id)
    remote_id = client.create_experiment(
        "Evolucio/test",
        artifact_location=(tmp_path / "mlflow-artifacts").as_uri(),
        tags={"evolucio.experiment_id": canonical_id, "evolucio.project": "Evolucio"},
    )
    return canonical_id, remote_id
