"""Validated portable dataset indexes; only indexed parts are analytical inputs."""

from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evolucio.persistence.local.artifacts import artifact_path, atomic_write_text, json_text

from .schemas import PARQUET_DATASET_SCHEMA_NAME, SCHEMAS, ParquetPersistencePolicy

INDEX_PATH = "artifacts/parquet/datasets.json"


class IndexModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Part(IndexModel):
    relative_path: str
    chunk_index: int = Field(ge=0)
    row_count: int = Field(gt=0)
    byte_size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    min_step: int
    max_step: int


class Dataset(IndexModel):
    schema_version: Literal[1] = 1
    relative_directory: str
    row_count: int = 0
    part_count: int = 0
    min_step: int | None = None
    max_step: int | None = None
    parts: list[Part] = Field(default_factory=lambda: list[Part]())


class DatasetIndex(IndexModel):
    schema_name: Literal["run_telemetry_parquet_v1"] = PARQUET_DATASET_SCHEMA_NAME
    schema_version: Literal[1] = 1
    run_id: str
    policy: ParquetPersistencePolicy
    complete: bool = False
    datasets: dict[str, Dataset]


def load_index(run_dir: Path) -> DatasetIndex:
    index = DatasetIndex.model_validate_json(artifact_path(run_dir, INDEX_PATH).read_text())
    if set(index.datasets) != set(SCHEMAS):
        raise ValueError("incompatible dataset names")
    for name, dataset in index.datasets.items():
        directory = f"artifacts/parquet/{name}"
        if dataset.relative_directory != directory:
            raise ValueError("invalid dataset directory")
        seen: set[int] = set()
        for part in dataset.parts:
            if part.relative_path != f"{directory}/part-{part.chunk_index:08d}.parquet":
                raise ValueError("invalid part path")
            artifact_path(run_dir, part.relative_path)
            if part.chunk_index in seen:
                raise ValueError("duplicate chunk part")
            seen.add(part.chunk_index)
    return index


def save_index(run_dir: Path, index: DatasetIndex) -> None:
    payload = index.model_dump()
    payload["policy"] = asdict(index.policy)
    atomic_write_text(artifact_path(run_dir, INDEX_PATH), json_text(payload))
