"""Projection of canonical scalar telemetry and synchronous MLflow batches."""

from collections.abc import Iterable, Iterator, Mapping
from itertools import batched
from pathlib import Path

import polars as pl
from mlflow import MlflowClient
from mlflow.entities import Metric

from evolucio.host import ChunkTelemetryDTO
from evolucio.persistence.local.artifacts import artifact_path, sha256_file
from evolucio.persistence.parquet.index import INDEX_PATH, load_index

from .errors import MlflowTrackingIntegrityError
from .types import MlflowTrackingPolicy

# MLflow Tracking REST API limit per log-batch request.
METRICS_PER_BATCH = 1000
STEP_METRICS = {
    "alive_count": "population.alive",
    "birth_count": "population.births",
    "death_count": "population.deaths",
    "total_energy_alive": "energy.total_alive",
    "total_resources": "resources.total",
    "active_lineage_count": "lineages.active",
    "genetic_parameter_variance": "genetics.parameter_variance",
    "mean_environment": "environment.mean",
}
FINAL_METRICS = {
    "alive_count": "final.alive_count",
    "births_total": "final.births_total",
    "deaths_total": "final.deaths_total",
    "active_lineage_count": "final.active_lineages",
    "final_total_energy": "final.total_energy",
    "final_total_resources": "final.total_resources",
    "agents_total": "final.agent_summary_rows",
    "lineages_total": "final.lineage_summary_rows",
    "elapsed_seconds": "runtime.elapsed_seconds",
    "steps_per_second": "runtime.steps_per_second",
}


def read_temporal_metrics(
    run_dir: Path, run_id: str, interval: int, timestamp: int
) -> Iterator[Metric]:
    if not artifact_path(run_dir, INDEX_PATH).is_file():
        return
    index = load_index(run_dir)
    if index.run_id != run_id:
        raise MlflowTrackingIntegrityError("Parquet index run identity differs")
    for part in index.datasets["metrics"].parts:
        path = artifact_path(run_dir, part.relative_path)
        if sha256_file(path) != part.sha256:
            raise MlflowTrackingIntegrityError("Parquet metrics part digest differs")
        rows = (
            pl.scan_parquet(path)
            .select("step", *STEP_METRICS)
            .filter(pl.col("step") % interval == 0)
            .collect(engine="streaming")
        )
        for step, *values in rows.iter_rows():
            for name, value in zip(STEP_METRICS.values(), values, strict=True):
                yield Metric(name, float(value), timestamp, step)


def final_metrics(summary: Mapping[str, object], step: int, timestamp: int) -> Iterator[Metric]:
    for source, name in FINAL_METRICS.items():
        value = summary.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            yield Metric(name, float(value), timestamp, step)


def log_metrics(client: MlflowClient, run_id: str, metrics: Iterable[Metric]) -> None:
    for batch in batched(metrics, METRICS_PER_BATCH):
        client.log_batch(run_id, metrics=batch, synchronous=True)


class MlflowChunkConsumer:
    def __init__(
        self, client: MlflowClient, run_id: str, policy: MlflowTrackingPolicy, timestamp: int
    ) -> None:
        self.client = client
        self.run_id = run_id
        self.policy = policy
        self.timestamp = timestamp

    def __call__(self, telemetry: ChunkTelemetryDTO) -> None:
        log_metrics(
            self.client,
            self.run_id,
            (
                Metric(key, float(getattr(row, source)), self.timestamp, row.step)
                for row in telemetry.metrics
                if row.step % self.policy.metrics_interval == 0
                for source, key in STEP_METRICS.items()
            ),
        )
