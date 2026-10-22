# ruff: noqa: ANN001, ANN201
from unittest.mock import patch

import pytest

from evolucio.persistence.parquet import (
    ParquetPersistencePolicy,
    ParquetRunConsumer,
    scan_births,
    scan_deaths,
    scan_genealogy,
    scan_metrics,
    verify_parquet_datasets,
)
from evolucio.persistence.parquet.index import load_index

from .conftest import birth, chunk, death


def test_events_never_sampled_and_terminal(recorder):
    sink = ParquetRunConsumer(recorder, ParquetPersistencePolicy(metrics_interval=10))
    steps = [3, 7, 11, 18]
    sink(
        chunk(
            births=[birth(s, s + 100, 1) for s in steps], deaths=[death(s, s + 200) for s in steps]
        )
    )
    summary = sink.finalize()
    assert summary.complete and summary.birth_rows == summary.genealogy_rows == 4
    assert scan_metrics(recorder.run_dir).collect()["step"].to_list() == [0, 10]
    assert scan_births(recorder.run_dir).collect()["birth_step"].to_list() == steps
    assert scan_deaths(recorder.run_dir).collect()["death_step"].to_list() == steps
    assert scan_genealogy(recorder.run_dir).collect().height == 4
    assert verify_parquet_datasets(recorder.run_dir).ok
    with pytest.raises(ValueError):
        sink.finalize()
    with pytest.raises(ValueError):
        sink(chunk())


def test_partial_failure_orphans_and_readers(recorder):
    sink = ParquetRunConsumer(recorder)
    sink(chunk(0, 0, 4))
    sink(chunk(1, 4, 8))
    with (
        patch("evolucio.persistence.parquet.writer.os.replace", side_effect=OSError("disk")),
        pytest.raises(OSError),
    ):
        sink(chunk(2, 8, 12))
    sink.finalize_partial()
    assert not load_index(recorder.run_dir).complete
    assert scan_metrics(recorder.run_dir).collect().height == 8
    assert verify_parquet_datasets(recorder.run_dir).ok
    assert not list(recorder.run_dir.rglob("*.tmp"))
    part = next(recorder.run_dir.rglob("*.parquet"))
    orphan = part.with_name("part-00000009.parquet")
    orphan.write_bytes(part.read_bytes())
    assert verify_parquet_datasets(recorder.run_dir).orphan_parts
    assert scan_metrics(recorder.run_dir).collect().height == 8


def test_damage_missing_and_schema(recorder):
    sink = ParquetRunConsumer(recorder)
    sink(chunk())
    sink.finalize()
    part = next(recorder.run_dir.rglob("*.parquet"))
    part.write_bytes(b"damaged")
    assert verify_parquet_datasets(recorder.run_dir).hash_mismatches
    assert part.read_bytes() == b"damaged"
    part.unlink()
    assert verify_parquet_datasets(recorder.run_dir).missing_parts


def test_duplicate_across_chunks_prevents_completion(recorder):
    sink = ParquetRunConsumer(recorder)
    sink(chunk(0, 0, 4, [birth(1, 2, 1)]))
    sink(chunk(1, 4, 8, [birth(5, 2, 1)]))
    with pytest.raises(ValueError, match="duplicate"):
        sink.finalize()
    sink.finalize_partial()


def test_failure_between_parts_and_index_commit(recorder):
    sink = ParquetRunConsumer(recorder)
    sink(chunk(0, 0, 4))
    with (
        patch("evolucio.persistence.parquet.writer.save_index", side_effect=OSError("index")),
        pytest.raises(OSError),
    ):
        sink(chunk(1, 4, 8))
    with pytest.raises(ValueError):
        sink.finalize()
    sink.finalize_partial()
    report = verify_parquet_datasets(recorder.run_dir)
    assert not report.ok and len(report.orphan_parts) == 1
    assert scan_metrics(recorder.run_dir).collect().height == 4


def test_empty_sampled_chunk_and_compression(recorder, monkeypatch):
    import polars as pl

    original = pl.DataFrame.write_parquet
    compressions = []

    def capture(self, path, *, compression) -> None:
        compressions.append(compression)
        original(self, path, compression=compression)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", capture)
    sink = ParquetRunConsumer(
        recorder, ParquetPersistencePolicy(metrics_interval=10, compression="gzip")
    )
    sink(chunk(0, 1, 4, [birth(2, 2, 1)]))
    sink.finalize()
    assert compressions == ["gzip", "gzip"]
    assert not load_index(recorder.run_dir).datasets["metrics"].parts
