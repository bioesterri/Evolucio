# ruff: noqa: ANN001, ANN201
from unittest.mock import patch

import polars as pl
import pytest
from typer.testing import CliRunner

from evolucio.cli.main import app
from evolucio.config import load_config
from evolucio.persistence.local import load_run_manifest, verify_local_run
from evolucio.persistence.parquet import scan_births, scan_deaths, scan_genealogy, scan_metrics
from evolucio.persistence.parquet.index import load_index
from evolucio.runtime import RunPlan
from evolucio.runtime.recorded_run import run_recorded_simulation
from tests.experiment_helpers import write_cases


def test_real_run_queries_and_tampering(tmp_path):
    config = load_config(write_cases(tmp_path)[0].config_path)
    run = run_recorded_simulation(config=config, plan=RunPlan(16, 4), output_root=tmp_path / "runs")
    directory = run.run_dir
    assert verify_local_run(directory).ok
    assert scan_metrics(directory).select(
        pl.len(), pl.col("step").min().alias("min"), pl.col("step").max().alias("max")
    ).collect().row(0) == (16, 0, 15)
    scan_births(directory).select(pl.len()).collect()
    scan_deaths(directory).group_by("cause_code").len().collect()
    scan_genealogy(directory).group_by("parent_agent_id").len().collect()
    part = next(directory.rglob("*.parquet"))
    part.write_bytes(part.read_bytes() + b"x")
    verification = verify_local_run(directory)
    assert not verification.ok
    assert str(part.relative_to(directory)) in verification.hash_mismatches
    result = CliRunner().invoke(app, ["artifacts", "verify", str(directory)])
    assert result.exit_code != 0


def test_runtime_failure_preserves_two_chunks(tmp_path):
    config = load_config(write_cases(tmp_path)[0].config_path)
    from evolucio.persistence.parquet.writer import ParquetRunConsumer

    original = ParquetRunConsumer.__call__

    def failing(self, batch) -> None:
        if batch.chunk_index == 2:
            raise OSError("controlled failure")
        original(self, batch)

    with patch.object(ParquetRunConsumer, "__call__", failing), pytest.raises(OSError):
        run_recorded_simulation(config=config, plan=RunPlan(16, 4), output_root=tmp_path / "runs")
    directory = next((tmp_path / "runs").iterdir())
    assert load_run_manifest(directory / "manifest.json").status == "failed"
    assert not load_index(directory).complete
    assert scan_metrics(directory).collect().height == 8
    assert verify_local_run(directory).ok
