# ruff: noqa: ANN001, ANN201
from dataclasses import replace

import polars as pl
import pytest

from evolucio.persistence.local.artifacts import sha256_file
from evolucio.persistence.parquet import ParquetRunConsumer, verify_parquet_datasets
from evolucio.persistence.parquet.frames import births_frame, deaths_frame
from evolucio.persistence.parquet.index import INDEX_PATH, load_index, save_index
from evolucio.persistence.parquet.writer import write_part

from .conftest import birth, chunk, death


def rewrite(recorder, dataset, transform):
    index = load_index(recorder.run_dir)
    part = index.datasets[dataset].parts[0]
    path = recorder.run_dir / part.relative_path
    transform(pl.read_parquet(path)).write_parquet(path)
    part.sha256 = sha256_file(path)
    part.byte_size = path.stat().st_size
    save_index(recorder.run_dir, index)


@pytest.mark.parametrize("change", ["schema", "counts", "projection", "sample", "identity"])
def test_verification_layers(recorder, change):
    sink = ParquetRunConsumer(recorder)
    sink(chunk(births=[birth(1, 2, 1)], deaths=[death(3, 2)]))
    if change == "schema":
        rewrite(recorder, "metrics", lambda f: f.with_columns(pl.col("step").cast(pl.Int32)))
        assert verify_parquet_datasets(recorder.run_dir).schema_mismatches
    elif change == "counts":
        index = load_index(recorder.run_dir)
        index.datasets["births"].row_count = 100
        save_index(recorder.run_dir, index)
        assert verify_parquet_datasets(recorder.run_dir).consistency_errors
    elif change == "projection":
        rewrite(
            recorder,
            "genealogy",
            lambda f: f.with_columns((pl.col("parent_agent_id") + 1).alias("parent_agent_id")),
        )
        assert "projection" in str(verify_parquet_datasets(recorder.run_dir).consistency_errors)
    elif change == "sample":
        index = load_index(recorder.run_dir)
        index.policy = replace(index.policy, metrics_interval=10)
        save_index(recorder.run_dir, index)
        assert "sampling" in str(verify_parquet_datasets(recorder.run_dir).consistency_errors)
    else:
        rewrite(recorder, "deaths", lambda f: f.with_columns(pl.lit("wrong").alias("run_id")))
        assert "run_id" in str(verify_parquet_datasets(recorder.run_dir).consistency_errors)
    with pytest.raises(ValueError):
        sink.finalize()


def test_death_before_birth(recorder):
    sink = ParquetRunConsumer(recorder)
    sink(chunk(births=[birth(4, 2, 1)], deaths=[death(1, 2)]))
    assert "precedes" in str(verify_parquet_datasets(recorder.run_dir).consistency_errors)


def test_sentinel_and_codes_rejected():
    with pytest.raises(ValueError, match="sentinel"):
        births_frame("r", [birth(1, -1, 1)])
    with pytest.raises(ValueError, match="cause"):
        deaths_frame("r", [replace(death(1, 1), cause_code=0)])
    with pytest.raises(ValueError, match="phase"):
        deaths_frame("r", [replace(death(1, 1), phase_code=0)])


def test_immutable_part_and_relative_verification(recorder, monkeypatch):
    sink = ParquetRunConsumer(recorder)
    sink(chunk())
    sink.finalize()
    part = next(recorder.run_dir.rglob("*.parquet"))
    data = part.read_bytes()
    with pytest.raises(FileExistsError):
        write_part(part, pl.DataFrame({"a": [1]}), "zstd")
    assert part.read_bytes() == data
    monkeypatch.chdir(recorder.run_dir.parent)
    from pathlib import Path

    assert verify_parquet_datasets(Path(recorder.run_id)).ok
    assert (recorder.run_dir / INDEX_PATH).is_file()
