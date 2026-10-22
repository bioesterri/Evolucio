# ruff: noqa: ANN001, ANN201
import json
import shutil
from unittest.mock import patch

import equinox as eqx
import jax
import pytest
from typer.testing import CliRunner

from evolucio.cli.main import app
from evolucio.config import load_config
from evolucio.persistence.local import load_run_manifest, verify_local_run
from evolucio.persistence.zarr import load_checkpoint, load_snapshot
from evolucio.persistence.zarr.index import latest_checkpoint, load_index
from evolucio.persistence.zarr.schemas import ZarrCapturePolicy
from evolucio.runtime import RunMode, RunPlan
from evolucio.runtime.recorded_run import run_recorded_simulation
from tests.experiment_helpers import write_cases
from tests.regression.helpers import assert_pytree_equivalent


def test_recorded_captures_coexist_with_parquet_and_new_run_continuation(tmp_path):
    config = load_config(write_cases(tmp_path)[0].config_path)
    policy = ZarrCapturePolicy(
        checkpoint_interval=10,
        snapshot_interval=5,
        checkpoint_on_completion=True,
        snapshot_on_completion=True,
    )
    whole = run_recorded_simulation(
        config=config, plan=RunPlan(20, 7), capture_policy=policy, output_root=tmp_path / "runs"
    )
    index = load_index(whole.run_dir)
    assert [r.state_step for r in index.checkpoints] == [10, 20]
    assert [r.state_step for r in index.snapshots] == [5, 10, 15, 20]
    assert index.complete and verify_local_run(whole.run_dir).ok
    assert {a.kind for a in whole.manifest.artifacts} == {
        "configuration",
        "summary",
        "parquet_dataset_index",
        "zarr_artifact_index",
    }
    assert len(whole.manifest.artifacts) == 4
    path = whole.run_dir / index.checkpoints[0].relative_path
    assert load_checkpoint(checkpoint_path=path).state_step == 10
    shutil.rmtree(whole.run_dir / "artifacts/zarr/snapshots")
    assert load_checkpoint(checkpoint_path=path).state_step == 10
    continuation = run_recorded_simulation(
        config=config,
        plan=RunPlan(10, 4),
        source_checkpoint=path,
        capture_policy=ZarrCapturePolicy(checkpoint_interval=10, snapshot_interval=6),
        output_root=tmp_path / "runs",
    )
    continued_index = load_index(continuation.run_dir)
    assert [r.state_step for r in continued_index.checkpoints] == [20]
    assert [r.state_step for r in continued_index.snapshots] == [12, 18]
    assert continuation.manifest.initial_step == 10
    assert continuation.manifest.final_step == 20
    assert continuation.manifest.run_id != whole.manifest.run_id
    assert verify_local_run(continuation.run_dir).ok
    provenance = json.loads((continuation.run_dir / "artifacts/checkpoint_source.json").read_text())
    assert provenance == {
        "source_run_id": whole.manifest.run_id,
        "source_checkpoint_step": 10,
        "source_checkpoint_hash": index.checkpoints[0].sha256,
    }
    assert_pytree_equivalent(continuation.result.runtime.state, whole.result.runtime.state)
    assert_pytree_equivalent(
        continuation.result.runtime.accumulator, whole.result.runtime.accumulator
    )


@pytest.mark.parametrize(
    "mode", [RunMode.EXPERIMENT, RunMode.SMOKE, RunMode.DEBUG, RunMode.BENCHMARK]
)
def test_capture_boundaries_support_all_modes_and_final_only(tmp_path, mode):
    config = load_config(write_cases(tmp_path)[0].config_path)
    run = run_recorded_simulation(
        config=config,
        plan=RunPlan(4, 3),
        mode=mode,
        capture_policy=ZarrCapturePolicy(
            checkpoint_on_completion=True, snapshot_on_completion=True
        ),
        output_root=tmp_path / "runs",
    )
    index = load_index(run.run_dir)
    assert [r.state_step for r in index.checkpoints] == [4]
    assert [r.state_step for r in index.snapshots] == [4]
    assert verify_local_run(run.run_dir).ok


def test_failure_keeps_completed_captures(tmp_path):
    config = load_config(write_cases(tmp_path)[0].config_path)
    from evolucio.runtime.capture_runner import RunCaptures

    original = RunCaptures.capture

    def fail(self, state, accumulator, *, final) -> None:
        if int(state.step) > 10:
            raise OSError("capture failure")
        original(self, state, accumulator, final=final)

    with patch.object(RunCaptures, "capture", fail), pytest.raises(OSError):
        run_recorded_simulation(
            config=config,
            plan=RunPlan(20, 7),
            capture_policy=ZarrCapturePolicy(checkpoint_interval=10, snapshot_interval=5),
            output_root=tmp_path / "runs",
        )
    directory = next((tmp_path / "runs").iterdir())
    assert load_run_manifest(directory / "manifest.json").status == "failed"
    index = load_index(directory)
    assert not index.complete
    assert [r.state_step for r in index.checkpoints] == [10]
    assert [r.state_step for r in index.snapshots] == [5, 10]
    assert verify_local_run(directory).ok
    assert (
        load_checkpoint(checkpoint_path=directory / index.checkpoints[0].relative_path).state_step
        == 10
    )


def test_cli_dry_run_and_read_only_commands(tmp_path, monkeypatch):
    config_path = write_cases(tmp_path)[0].config_path
    from unittest.mock import Mock

    compile_config = Mock(side_effect=AssertionError("dry run must not compile"))
    with patch("evolucio.config.compile.compile_config", compile_config):
        result = CliRunner().invoke(
            app,
            [
                "run",
                str(config_path),
                "--steps",
                "20",
                "--chunk-steps",
                "7",
                "--checkpoint-every",
                "10",
                "--snapshot-every",
                "5",
                "--dry-run",
                "--output-root",
                str(tmp_path / "dry"),
            ],
        )
    assert result.exit_code == 0, result.output
    assert "chunks: 4" in result.output
    assert "planned_chunk_lengths: [5, 5, 5, 5]" in result.output
    assert "expected_checkpoint_steps: [10, 20]" in result.output
    assert "expected_snapshot_steps: [5, 10, 15, 20]" in result.output
    assert not (tmp_path / "dry").exists()
    config = load_config(config_path)
    run = run_recorded_simulation(
        config=config,
        plan=RunPlan(2, 2),
        capture_policy=ZarrCapturePolicy(
            checkpoint_on_completion=True, snapshot_on_completion=True
        ),
        output_root=tmp_path / "runs",
    )
    cp = latest_checkpoint(run.run_dir)
    path = run.run_dir / cp.relative_path
    assert CliRunner().invoke(app, ["checkpoint", "list", str(run.run_dir)]).exit_code == 0
    assert CliRunner().invoke(app, ["snapshot", "list", str(run.run_dir)]).exit_code == 0
    assert CliRunner().invoke(app, ["checkpoint", "verify", str(path)]).exit_code == 0
    assert CliRunner().invoke(app, ["artifacts", "verify", str(run.run_dir)]).exit_code == 0
    internal = next(p for p in path.rglob("*") if p.is_file() and "c" in p.relative_to(path).parts)
    internal.write_bytes(b"corrupt")
    assert not verify_local_run(run.run_dir).ok
    assert CliRunner().invoke(app, ["checkpoint", "verify", str(path)]).exit_code != 0
    assert CliRunner().invoke(app, ["artifacts", "verify", str(run.run_dir)]).exit_code != 0


def test_extinction_snapshot_compact_checkpoint_full(tmp_path):
    config = load_config(write_cases(tmp_path)[0].config_path)
    run = run_recorded_simulation(
        config=config,
        plan=RunPlan(10, 7),
        capture_policy=ZarrCapturePolicy(
            checkpoint_on_completion=True, snapshot_on_completion=True
        ),
        output_root=tmp_path / "runs",
    )
    index = load_index(run.run_dir)
    state = load_checkpoint(checkpoint_path=run.run_dir / index.checkpoints[0].relative_path).state
    snapshot = load_snapshot(run.run_dir / index.snapshots[0].relative_path)
    assert not state.population.alive.any()
    assert state.population.alive.shape == (8,)
    assert state.genomes.layer1.weight.shape == (8, 16, 15)
    assert snapshot.population.position.shape == (0, 2)
    assert snapshot.world.occupancy.shape == (4, 4)


@pytest.fixture(scope="module", autouse=True)
def release_test_compilation_caches():
    """Release scenario-specific executables before the rest of the suite runs."""
    yield
    jax.clear_caches()
    eqx.clear_caches()
