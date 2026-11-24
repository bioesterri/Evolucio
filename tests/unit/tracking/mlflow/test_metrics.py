"""Scalar projection and absolute sampling using a real MLflow SQLite backend."""

from pathlib import Path

import pytest
from mlflow import MlflowClient
from mlflow.entities import Param

from evolucio.config import load_config
from evolucio.host import ChunkTelemetryDTO, EventBatchDTO, StepMetricsDTO
from evolucio.host.schema import HOST_DTO_SCHEMA_VERSION
from evolucio.persistence.local import create_local_run
from evolucio.runtime import RunMode, RunPlan
from evolucio.tracking.mlflow.errors import MlflowTrackingIntegrityError
from evolucio.tracking.mlflow.metrics import STEP_METRICS, MlflowChunkConsumer
from evolucio.tracking.mlflow.params import build_mlflow_params
from evolucio.tracking.mlflow.runs import log_inputs
from evolucio.tracking.mlflow.types import MlflowTrackingPolicy


def test_absolute_sampling_and_chunk_neutrality(tmp_path: Path) -> None:
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'tracking.sqlite'}")
    experiment = client.create_experiment(
        "test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    histories = []
    for chunk_length in (4, 7):
        run_id = client.create_run(experiment).info.run_id
        consumer = MlflowChunkConsumer(client, run_id, MlflowTrackingPolicy(2), 123)
        for chunk, start in enumerate(range(0, 9, chunk_length)):
            stop = min(start + chunk_length, 9)
            metrics = tuple(
                StepMetricsDTO(
                    step,
                    step + 1,
                    0,
                    0,
                    0,
                    (),
                    8.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0,
                    0,
                    0,
                    1,
                    0.25,
                    1.0,
                )
                for step in range(start, stop)
            )
            consumer(
                ChunkTelemetryDTO(
                    HOST_DTO_SCHEMA_VERSION, chunk, start, stop, metrics, EventBatchDTO((), ())
                )
            )
        histories.append(
            {
                name: [(p.step, p.value) for p in client.get_metric_history(run_id, name)]
                for name in STEP_METRICS.values()
            }
        )
    assert histories[0] == histories[1]
    assert histories[0]["population.alive"] == [(step, step + 1) for step in (0, 2, 4, 6, 8)]
    with pytest.raises(ValueError, match="positive"):
        MlflowTrackingPolicy(0)


def test_params_are_immutable_and_scalar(tmp_path: Path) -> None:
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'tracking.sqlite'}")
    experiment = client.create_experiment(
        "test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    remote = client.create_run(experiment)
    run_id = remote.info.run_id
    config = load_config("tests/fixtures/config/valid_v1.json")
    recorder = create_local_run(
        config=config, plan=RunPlan(7, 4), mode=RunMode.EXPERIMENT, output_root=tmp_path
    )
    params = build_mlflow_params(config, recorder.manifest)
    assert params["total_steps"] == "7" and params["chunk_steps"] == "4"
    assert "world.environment_schedule" not in params
    client.log_batch(run_id, params=[Param("seed", str(config.seed + 1))])
    with pytest.raises(MlflowTrackingIntegrityError, match="seed"):
        log_inputs(client, client.get_run(run_id), params, {})
    assert "total_steps" not in client.get_run(run_id).data.params
