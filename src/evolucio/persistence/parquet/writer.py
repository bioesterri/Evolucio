"""Synchronous single-writer chunk persistence with immutable part files."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from evolucio.host import HOST_DTO_SCHEMA_VERSION, ChunkTelemetryDTO
from evolucio.persistence.local.artifacts import artifact_path, sha256_file
from evolucio.persistence.local.recorder import LocalRunRecorder
from evolucio.persistence.parquet.schemas import DEFAULT_PARQUET_POLICY

from .frames import births_frame, deaths_frame, genealogy_frame, metrics_frame
from .index import INDEX_PATH, Dataset, DatasetIndex, Part, save_index
from .schemas import SCHEMAS, STEP_COLUMNS, ParquetCompression, ParquetPersistencePolicy
from .verify import verify_parquet_datasets


@dataclass(frozen=True, slots=True)
class ParquetDatasetSummary:
    complete: bool
    metrics_rows: int
    birth_rows: int
    death_rows: int
    genealogy_rows: int
    parquet_dataset_index: str = INDEX_PATH


def write_part(path: Path, frame: pl.DataFrame, compression: ParquetCompression) -> None:
    if path.exists():
        raise FileExistsError(f"immutable part already exists: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".part-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        frame.write_parquet(temporary, compression=compression)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class ParquetRunConsumer:
    """Retain only part metadata; failed chunks never enter the committed index."""

    def __init__(
        self,
        recorder: LocalRunRecorder,
        policy: ParquetPersistencePolicy = DEFAULT_PARQUET_POLICY,
    ) -> None:
        self.recorder = recorder
        self.policy = policy
        self.closed = False
        self.failed = False
        self.last_chunk = -1
        self.last_end: int | None = None
        root = artifact_path(recorder.run_dir, "artifacts/parquet")
        root.mkdir(exist_ok=False)
        datasets: dict[str, Dataset] = {}
        for name in SCHEMAS:
            directory = f"artifacts/parquet/{name}"
            artifact_path(recorder.run_dir, directory).mkdir()
            datasets[name] = Dataset(relative_directory=directory)
        self.index = DatasetIndex(run_id=recorder.run_id, policy=policy, datasets=datasets)
        save_index(recorder.run_dir, self.index)

    def __call__(self, batch: ChunkTelemetryDTO) -> None:
        if self.closed or self.failed:
            raise ValueError("consumer is closed or failed")
        if (
            batch.schema_version != HOST_DTO_SCHEMA_VERSION
            or batch.chunk_index <= self.last_chunk
            or batch.start_step >= batch.end_step_exclusive
            or (self.last_end is not None and batch.start_step != self.last_end)
        ):
            raise ValueError("invalid telemetry chunk sequence")
        for rows, temporal in (
            (batch.metrics, "step"),
            (batch.events.births, "birth_step"),
            (batch.events.deaths, "death_step"),
        ):
            if any(
                not batch.start_step <= getattr(row, temporal) < batch.end_step_exclusive
                for row in rows
            ):
                raise ValueError("telemetry outside chunk interval")
        frames = {
            "metrics": metrics_frame(self.recorder.run_id, batch.metrics, self.policy),
            "births": births_frame(self.recorder.run_id, batch.events.births),
            "deaths": deaths_frame(self.recorder.run_id, batch.events.deaths),
            "genealogy": genealogy_frame(self.recorder.run_id, batch.events.births),
        }
        candidate = self.index.model_copy(deep=True)
        try:
            for name, frame in frames.items():
                if not frame.height:
                    continue
                dataset = candidate.datasets[name]
                relative = f"{dataset.relative_directory}/part-{batch.chunk_index:08d}.parquet"
                path = artifact_path(self.recorder.run_dir, relative)
                write_part(path, frame, self.policy.compression)
                steps = frame[STEP_COLUMNS[name]]
                low, high = int(steps[0]), int(steps[-1])
                dataset.parts.append(
                    Part(
                        relative_path=relative,
                        chunk_index=batch.chunk_index,
                        row_count=frame.height,
                        byte_size=path.stat().st_size,
                        sha256=sha256_file(path),
                        min_step=low,
                        max_step=high,
                    )
                )
                dataset.row_count += frame.height
                dataset.part_count += 1
                dataset.min_step = low if dataset.min_step is None else min(dataset.min_step, low)
                dataset.max_step = high if dataset.max_step is None else max(dataset.max_step, high)
            save_index(self.recorder.run_dir, candidate)
        except Exception:
            self.failed = True
            raise
        self.index = candidate
        self.last_chunk = batch.chunk_index
        self.last_end = batch.end_step_exclusive

    def _finish(self, complete: bool) -> ParquetDatasetSummary:
        if self.closed:
            raise ValueError("consumer already finalized")
        if complete and self.failed:
            raise ValueError("failed consumer cannot complete")
        if complete:
            report = verify_parquet_datasets(self.recorder.run_dir)
            if not report.ok:
                raise ValueError(f"Parquet integrity verification failed: {report}")
        self.index.complete = complete
        save_index(self.recorder.run_dir, self.index)
        self.recorder.register_artifact(
            name="parquet_dataset_index",
            kind="parquet_dataset_index",
            relative_path=INDEX_PATH,
            media_type="application/json",
        )
        self.closed = True
        counts = self.index.datasets
        return ParquetDatasetSummary(
            complete,
            counts["metrics"].row_count,
            counts["births"].row_count,
            counts["deaths"].row_count,
            counts["genealogy"].row_count,
        )

    def finalize(self) -> ParquetDatasetSummary:
        return self._finish(True)

    def finalize_partial(self) -> ParquetDatasetSummary:
        return self._finish(False)
