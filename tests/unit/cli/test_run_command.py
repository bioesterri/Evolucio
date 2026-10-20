# ruff: noqa: ANN001, ANN201

import json
from unittest.mock import Mock

import pytest

from evolucio.cli.main import main


def summary(text):
    return {
        name: json.loads(value)
        for name, value in (line.split(": ", 1) for line in text.splitlines())
        if name not in ("run_id", "run_dir", "manifest")
    }


def test_invalid_does_not_run(config_path, monkeypatch, capsys):
    runtime = Mock()
    initialize = Mock()
    monkeypatch.setattr("evolucio.runtime.run_simulation", runtime)
    monkeypatch.setattr("evolucio.runtime.initialization.initialize_simulation", initialize)
    config_path.write_text("{}")
    assert main(["run", str(config_path), "--steps", "10"]) == 2
    runtime.assert_not_called()
    initialize.assert_not_called()
    assert capsys.readouterr().err


def test_dry_run(config_path, monkeypatch, capsys):
    runtime = Mock()
    initialize = Mock()
    compile_config = Mock()
    monkeypatch.setattr("evolucio.runtime.run_simulation", runtime)
    monkeypatch.setattr("evolucio.runtime.initialization.initialize_simulation", initialize)
    monkeypatch.setattr("evolucio.config.compile.compile_config", compile_config)
    assert main(["run", str(config_path), "--steps", "10", "--chunk-steps", "4", "--dry-run"]) == 0
    result = summary(capsys.readouterr().out)
    assert result["total_steps"] == 10
    assert result["chunk_steps"] == 4
    assert result["chunks"] == 3
    assert result["full_chunks"] == 2
    assert result["remainder"] == 2
    assert result["seed"] == 42
    runtime.assert_not_called()
    initialize.assert_not_called()
    compile_config.assert_not_called()


def test_default_chunk(config_path, capsys):
    assert main(["run", str(config_path), "--steps", "1", "--dry-run"]) == 0
    assert summary(capsys.readouterr().out)["chunk_steps"] == 4


@pytest.mark.parametrize(
    "options", [[], ["--steps", "0"], ["--steps", "true"], ["--steps", "1", "--chunk-steps", "-1"]]
)
def test_bad_plan(config_path, options, capsys):
    assert main(["run", str(config_path), *options]) == 2
    assert capsys.readouterr().err


@pytest.mark.parametrize("steps,chunk", [(1, 1), (5, 2)])
def test_real_run(config_path, capsys, steps, chunk):
    assert main(["run", str(config_path), "--steps", str(steps), "--chunk-steps", str(chunk)]) == 0
    output = capsys.readouterr()
    result = summary(output.out)
    assert result["status"] == "completed"
    assert result["steps_executed"] == result["final_step"] == steps
    assert result["alive_count"] == 2
    assert result["active_lineage_count"] == 2
    assert not output.err


def test_seed_and_chunk_neutrality(config_path, capsys):
    results = []
    for chunk in (4, 7, 4):
        assert main(["run", str(config_path), "--steps", "20", "--chunk-steps", str(chunk)]) == 0
        results.append(summary(capsys.readouterr().out))
    for result in results[1:]:
        assert result == pytest.approx(results[0])


@pytest.mark.parametrize("empty", [False, True])
def test_extinction_completes(config_path, config_data, capsys, empty):
    if empty:
        config_data["population"]["initial_agents"] = 0
    else:
        config_data["energy"].update(initial_energy=0.01, basal_cost=1.0)
    config_path.write_text(json.dumps(config_data))
    assert main(["run", str(config_path), "--steps", "2", "--chunk-steps", "1"]) == 0
    result = summary(capsys.readouterr().out)
    assert result["population_extinct"] is True
    assert result["final_step"] == 2
    assert result["deaths_total"] == (0 if empty else 2)


def test_quiet(config_path, capsys):
    assert main(["run", str(config_path), "--steps", "1", "--quiet"]) == 0
    assert not capsys.readouterr().out


def test_runtime_error(config_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "evolucio.runtime.initialization.initialize_simulation", Mock(return_value=object())
    )
    monkeypatch.setattr(
        "evolucio.runtime.run_simulation", Mock(side_effect=RuntimeError("test failure"))
    )
    assert main(["run", str(config_path), "--steps", "1"]) == 1
    output = capsys.readouterr()
    assert "RuntimeError: test failure" in output.err
    assert "run_id:" in output.out


def test_known_initialization_error(config_path, monkeypatch, capsys):
    from evolucio.runtime.initialization import InitializationError

    monkeypatch.setattr(
        "evolucio.runtime.initialization.initialize_simulation",
        Mock(side_effect=InitializationError("founder overflow")),
    )
    assert main(["run", str(config_path), "--steps", "1"]) == 1
    assert "founder overflow" in capsys.readouterr().err


def test_missing_telemetry(config_path, monkeypatch, capsys):
    monkeypatch.setattr("evolucio.runtime.initialization.initialize_simulation", Mock())
    monkeypatch.setattr("evolucio.runtime.run_simulation", Mock())
    assert main(["run", str(config_path), "--steps", "1"]) == 1
    assert 'status: "failed"' in capsys.readouterr().err


def test_consumer_retains_only_latest_record():
    from dataclasses import fields

    from evolucio.cli.run_command import CliRunSummaryConsumer
    from evolucio.host import ChunkTelemetryDTO, EventBatchDTO, StepMetricsDTO

    values = {field.name: 0 for field in fields(StepMetricsDTO)}
    values["deaths_by_cause"] = (0,) * 6
    first = StepMetricsDTO(**(values | {"birth_count": 2, "death_count": 1}))
    last = StepMetricsDTO(**(values | {"step": 1, "birth_count": 3, "death_count": 2}))
    consumer = CliRunSummaryConsumer()
    batch = ChunkTelemetryDTO(1, 0, 0, 2, (first, last), EventBatchDTO((), ()))
    consumer(batch)
    assert consumer.latest_metrics is last
    assert consumer.births_total == 5
    assert consumer.deaths_total == 3
    assert {field.name for field in fields(consumer)} == {
        "latest_metrics",
        "births_total",
        "deaths_total",
    }
