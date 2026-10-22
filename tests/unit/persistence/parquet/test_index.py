# ruff: noqa: ANN001, ANN201
import json
from dataclasses import replace

import pytest

from evolucio.persistence.parquet import ParquetRunConsumer, scan_births, verify_parquet_datasets
from evolucio.persistence.parquet.index import INDEX_PATH, load_index

from .conftest import chunk


@pytest.mark.parametrize("change", ["version", "directory", "names", "path", "duplicate"])
def test_reject_invalid_index(recorder, change):
    sink = ParquetRunConsumer(recorder)
    sink(chunk())
    path = recorder.run_dir / INDEX_PATH
    payload = json.loads(path.read_text())
    if change == "version":
        payload["schema_version"] = 2
    elif change == "directory":
        payload["datasets"]["metrics"]["relative_directory"] = "../escape"
    elif change == "names":
        del payload["datasets"]["deaths"]
    elif change == "path":
        payload["datasets"]["metrics"]["parts"][0]["relative_path"] = "../escape.parquet"
    else:
        parts = payload["datasets"]["metrics"]["parts"]
        parts.append(parts[0])
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_index(recorder.run_dir)
    assert not verify_parquet_datasets(recorder.run_dir).ok


def test_empty_reader_and_sequence(recorder):
    sink = ParquetRunConsumer(recorder)
    assert scan_births(recorder.run_dir).collect().height == 0
    with pytest.raises(ValueError, match="sequence"):
        sink(replace(chunk(), schema_version=999))
    with pytest.raises(ValueError, match="outside"):
        sink(replace(chunk(), end_step_exclusive=2))
    sink(chunk())
    with pytest.raises(ValueError):
        sink(chunk())
    sink.finalize()


def test_symlink_rejected(recorder, tmp_path):
    sink = ParquetRunConsumer(recorder)
    sink(chunk())
    part = next(recorder.run_dir.rglob("*.parquet"))
    target = tmp_path / "external.parquet"
    part.rename(target)
    part.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        load_index(recorder.run_dir)
    assert not verify_parquet_datasets(recorder.run_dir).ok
