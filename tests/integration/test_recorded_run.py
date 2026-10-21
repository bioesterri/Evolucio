# ruff: noqa: ANN001, ANN201, ANN202

import json
from uuid import UUID

import pytest

from evolucio.config import freeze_config, load_config
from evolucio.persistence.local import RunStatus, load_run_manifest, sha256_file, verify_local_run
from evolucio.runtime import RunMode, RunPlan
from evolucio.runtime.recorded_run import run_recorded_simulation
from tests.experiment_helpers import write_cases
from tests.regression.helpers import assert_pytree_equivalent


@pytest.mark.parametrize("mode", list(RunMode))
def test_audit_survives_loss_of_python_objects(tmp_path, monkeypatch, mode):
    from evolucio.runtime import initialization

    original = initialization.initialize_simulation
    before_hashes = []
    root = tmp_path / "runs"
    host = load_config(write_cases(tmp_path)[0].config_path)

    def initialize(config, *, seed):
        dirs = list(root.iterdir())
        assert len(dirs) == 1
        manifest = load_run_manifest(dirs[0] / "manifest.json")
        assert manifest.status == RunStatus.CREATED
        assert (dirs[0] / "config.json").exists()
        before_hashes.append(sha256_file(dirs[0] / "config.json"))
        return original(config, seed=seed)

    monkeypatch.setattr(initialization, "initialize_simulation", initialize)
    result = run_recorded_simulation(config=host, plan=RunPlan(4, 2), mode=mode, output_root=root)
    directory = result.run_dir
    assert result.manifest.status == RunStatus.COMPLETED
    del result
    manifest = load_run_manifest(directory / "manifest.json")
    config = load_config(directory / "config.json")
    summary = json.loads((directory / "summary.json").read_text())
    assert str(UUID(manifest.run_id)) == directory.name
    assert config == host and manifest.config_hash == freeze_config(config).config_hash
    assert manifest.seed == config.seed == summary["seed"]
    assert manifest.total_steps == manifest.steps_executed == summary["steps_executed"] == 4
    assert manifest.mode == mode and manifest.chunk_steps == 2
    assert manifest.code_version.package_version and manifest.environment.jax_version
    assert (
        manifest.schemas.simulation_step and manifest.schemas.metrics and manifest.schemas.host_dto
    )
    assert manifest.compile_signature
    assert manifest.finished_at_utc and manifest.error is None
    assert {a.relative_path for a in manifest.artifacts} == {
        "config.json",
        "summary.json",
        "artifacts/parquet/datasets.json",
    }
    for record in manifest.artifacts:
        assert record.sha256 == sha256_file(directory / record.relative_path)
        assert record.byte_size == (directory / record.relative_path).stat().st_size
    assert verify_local_run(directory).ok
    assert before_hashes == [sha256_file(directory / "config.json")]
    (directory / "config.json").write_bytes((directory / "config.json").read_bytes() + b"\n")
    assert not verify_local_run(directory).ok


@pytest.mark.parametrize("phase", ["initialization", "runtime", "consumer"])
def test_controlled_failure_is_retained_and_original_raised(tmp_path, monkeypatch, phase):
    from unittest.mock import Mock

    error = RuntimeError("controlled failure")
    consumer = None
    if phase == "initialization":
        monkeypatch.setattr(
            "evolucio.runtime.initialization.initialize_simulation", Mock(side_effect=error)
        )
    elif phase == "runtime":
        monkeypatch.setattr("evolucio.runtime.run_with_mode", Mock(side_effect=error))
    else:
        consumer = Mock(side_effect=error)
    config = load_config(write_cases(tmp_path)[0].config_path)
    with pytest.raises(RuntimeError) as caught:
        run_recorded_simulation(
            config=config, plan=RunPlan(4, 2), output_root=tmp_path / "runs", consumer=consumer
        )
    assert caught.value is error
    (directory,) = (tmp_path / "runs").iterdir()
    manifest = load_run_manifest(directory / "manifest.json")
    assert manifest.status == RunStatus.FAILED
    assert manifest.error.error_type == "RuntimeError"
    assert manifest.error.message == "controlled failure" and manifest.finished_at_utc
    assert (directory / "config.json").is_file() and (directory / "error.json").is_file()
    assert verify_local_run(directory).ok


def test_recording_does_not_change_scientific_trajectory(tmp_path):
    import jax
    import numpy as np

    from evolucio.config.compile import compile_config
    from evolucio.core.metrics import create_empty_metrics_accumulator
    from evolucio.runtime import run_with_mode
    from evolucio.runtime.initialization import initialize_simulation

    config = load_config(write_cases(tmp_path)[0].config_path)
    compiled = compile_config(config)
    reference = run_with_mode(
        state=initialize_simulation(compiled.core, seed=config.seed),
        accumulator=create_empty_metrics_accumulator(),
        config=compiled.core,
        plan=RunPlan(4, 2),
        mode=RunMode.EXPERIMENT,
    )
    recorded = run_recorded_simulation(
        config=config, compiled=compiled, plan=RunPlan(4, 2), output_root=tmp_path / "runs"
    )
    for name in ("population", "world", "genomes", "ids"):
        assert_pytree_equivalent(
            getattr(reference.runtime.state, name), getattr(recorded.result.runtime.state, name)
        )
    assert_pytree_equivalent(reference.runtime.accumulator, recorded.result.runtime.accumulator)
    np.testing.assert_array_equal(
        jax.random.key_data(reference.runtime.state.rng.key),
        jax.random.key_data(recorded.result.runtime.state.rng.key),
    )
    second = run_recorded_simulation(
        config=config, plan=RunPlan(4, 2), output_root=tmp_path / "runs"
    )
    assert recorded.manifest.run_id != second.manifest.run_id
    assert recorded.manifest.config_hash == second.manifest.config_hash
    assert recorded.summary == second.summary
    with pytest.raises(ValueError, match="does not match"):
        other = config.model_dump()
        other["seed"] += 1
        run_recorded_simulation(
            config=type(config).model_validate(other),
            compiled=compiled,
            plan=RunPlan(4, 2),
            output_root=tmp_path / "runs",
        )


def test_failure_recording_error_does_not_replace_original(tmp_path, monkeypatch):
    from unittest.mock import Mock

    error = RuntimeError("original simulation failure")
    monkeypatch.setattr(
        "evolucio.runtime.initialization.initialize_simulation", Mock(side_effect=error)
    )
    monkeypatch.setattr(
        "evolucio.persistence.local.LocalRunRecorder.mark_failed",
        Mock(side_effect=OSError("disk unavailable")),
    )
    config = load_config(write_cases(tmp_path)[0].config_path)
    with pytest.raises(RuntimeError) as caught:
        run_recorded_simulation(config=config, plan=RunPlan(2, 1), output_root=tmp_path / "runs")
    assert caught.value is error
    assert any("disk unavailable" in note for note in error.__notes__)


def test_debug_failure_preserves_step_and_structured_error(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from evolucio.runtime import DebugFailure, RunDiagnosticError

    error = RunDiagnosticError(RunMode.DEBUG, DebugFailure(2, "transition", ("TEST",), (1,)))
    monkeypatch.setattr("evolucio.runtime.run_with_mode", Mock(side_effect=error))
    config = load_config(write_cases(tmp_path)[0].config_path)
    with pytest.raises(RunDiagnosticError) as caught:
        run_recorded_simulation(
            config=config, plan=RunPlan(4, 2), mode=RunMode.DEBUG, output_root=tmp_path / "runs"
        )
    assert caught.value is error
    (directory,) = (tmp_path / "runs").iterdir()
    assert load_run_manifest(directory / "manifest.json").error.step == 2
