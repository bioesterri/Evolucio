"""Recovery preserves complete scientific trajectories and independent lifecycle records."""

from dataclasses import fields
from pathlib import Path

import pytest

from evolucio.analysis.population import population_series
from evolucio.analysis.summaries import AGENTS_PATH, LINEAGES_PATH
from evolucio.analysis.summary import analyze_run
from evolucio.host import ChunkTelemetryDTO
from evolucio.persistence.genomes import read_genome_index
from evolucio.persistence.loaders.run_data import open_run
from evolucio.persistence.local import LocalRunRecorder, RunStatus, load_run_manifest
from evolucio.persistence.local.version_info import CodeVersionInfo
from evolucio.persistence.parquet.reader import scan_dataset
from evolucio.persistence.validation.runner import validate_run
from evolucio.persistence.validation.types import CheckStatus, ValidationLevel
from evolucio.persistence.zarr.index import load_index
from evolucio.persistence.zarr.schemas import ZarrCapturePolicy
from evolucio.recovery.assessment import inspect_run
from evolucio.recovery.errors import CheckpointNotRecoverableError
from evolucio.recovery.provenance import read_provenance
from evolucio.recovery.runner import resume
from evolucio.runtime.recorded_run import RecordedRunResult, run_recorded_simulation
from evolucio.runtime.stop_control import StopController
from evolucio.runtime.types import HostChunkBatch, RunPlan
from tests.regression.helpers import assert_pytree_equivalent
from tests.zarr_helpers import checkpoint_host_config, checkpoint_scenario


def test_complete_trajectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polars as pl

    state, _ = checkpoint_scenario()
    identity = CodeVersionInfo("0.1.0.dev0", "a" * 40, False)

    def version() -> CodeVersionInfo:
        return identity

    monkeypatch.setattr("evolucio.persistence.local.version_info.code_version_info", version)
    monkeypatch.setattr("evolucio.recovery.assessment.code_version_info", version)
    monkeypatch.setattr(
        "evolucio.runtime.initialization.initialize_simulation", lambda *args, **kwargs: state
    )
    config = checkpoint_host_config()
    config = config.model_copy(
        update={
            "population": config.population.model_copy(update={"initial_agents": 5}),
        }
    )
    expected = run_recorded_simulation(config=config, plan=RunPlan(100, 10), output_root=tmp_path)
    controller = StopController()

    def stop_at_40(batch: HostChunkBatch) -> None:
        if batch.end_step_exclusive == 40:
            controller.request_stop()

    interrupted = run_recorded_simulation(
        config=config,
        plan=RunPlan(100, 10),
        output_root=tmp_path,
        capture_policy=ZarrCapturePolicy(checkpoint_interval=20),
        stop_controller=controller,
        consumer=stop_at_40,
    )
    assert interrupted.manifest.status is RunStatus.STOPPED
    assert interrupted.manifest.stop and interrupted.manifest.stop.ordered
    assert not (interrupted.run_dir / AGENTS_PATH).exists()
    assert inspect_run(interrupted.run_dir).latest_recoverable_checkpoint_step == 40
    files_before = {
        path.relative_to(interrupted.run_dir): path.read_bytes()
        for path in interrupted.run_dir.rglob("*")
        if path.is_file()
    }
    dry_run = resume(interrupted.run_dir, dry_run=True)
    assert not isinstance(dry_run, RecordedRunResult)
    assert dry_run.remaining_steps == 60
    result = resume(interrupted.run_dir, chunk_steps=7)
    assert isinstance(result, RecordedRunResult)
    assert result.manifest.status is RunStatus.COMPLETED
    assert result.manifest.run_id != interrupted.manifest.run_id
    assert result.summary.steps_executed == 60
    assert result.summary.final_step == 100
    provenance = read_provenance(result.run_dir, result.manifest)
    assert provenance and provenance.root_run_id == interrupted.manifest.run_id
    assert provenance.inherited_steps == 40
    for run in (expected, interrupted, result):
        validation = validate_run(
            run.run_dir, validation_level=ValidationLevel.SCIENTIFIC, allow_partial=True
        )
        assert validation.validation_passed, validation.issues
        if run is result:
            assert validation.components["recovery_source"] is CheckStatus.PASSED
    partial = analyze_run(open_run(interrupted.run_dir))
    assert partial.run_status == "stopped"
    assert partial.coverage.end_step_exclusive == 40
    assert partial.survival_summary is not None
    assert partial.survival_summary.right_censored_count == interrupted.summary.alive_count
    assert population_series(open_run(interrupted.run_dir)).collect()["step"].max() == 39
    continuous = analyze_run(open_run(expected.run_dir))
    recovered = analyze_run(open_run(result.run_dir))
    assert recovered.root_run_id == recovered.source_run_id == interrupted.manifest.run_id
    assert recovered.recovery_depth == 1
    assert recovered.population_summary == continuous.population_summary
    assert recovered.survival_summary == continuous.survival_summary
    assert recovered.mortality_summary.deaths_total == continuous.mortality_summary.deaths_total
    assert recovered.mortality_summary.by_cause == continuous.mortality_summary.by_cause
    assert recovered.mortality_summary.by_phase == continuous.mortality_summary.by_phase
    assert recovered.mortality_summary.observed_death_age == pytest.approx(
        continuous.mortality_summary.observed_death_age
    )
    for field in fields(recovered.energy_summary):
        actual = getattr(recovered.energy_summary, field.name)
        expected_metric = getattr(continuous.energy_summary, field.name)
        assert actual.value == pytest.approx(expected_metric.value, abs=3e-5), field.name
        assert actual.coverage == expected_metric.coverage
        assert actual.quality == expected_metric.quality
    assert_pytree_equivalent(result.result.runtime.state, expected.result.runtime.state)
    assert_pytree_equivalent(result.result.runtime.accumulator, expected.result.runtime.accumulator)
    for dataset in ("metrics", "births", "deaths", "genealogy"):
        assert (
            scan_dataset(result.run_dir, dataset)
            .collect()
            .drop("run_id")
            .equals(scan_dataset(expected.run_dir, dataset).collect().drop("run_id"))
        ), dataset
    for relative in (AGENTS_PATH, LINEAGES_PATH):
        assert (
            pl.read_parquet(result.run_dir / relative)
            .drop("run_id")
            .equals(pl.read_parquet(expected.run_dir / relative).drop("run_id"))
        )
    assert {
        record.genome_id: record.genome_hash for record in read_genome_index(result.run_dir).genomes
    } == {
        record.genome_id: record.genome_hash
        for record in read_genome_index(expected.run_dir).genomes
    }
    assert files_before == {
        path.relative_to(interrupted.run_dir): path.read_bytes()
        for path in interrupted.run_dir.rglob("*")
        if path.is_file()
    }
    latest = load_index(interrupted.run_dir).checkpoints[-1]
    (interrupted.run_dir / latest.relative_path / "zarr.json").write_text("{}")
    report = inspect_run(interrupted.run_dir)
    assert report.latest_recoverable_checkpoint_step == 20
    assert any("checkpoint 40" in error for error in report.integrity_errors)
    with pytest.raises(CheckpointNotRecoverableError):
        resume(interrupted.run_dir, checkpoint_step=40, dry_run=True)

    # Simulate a failure after a new, verified checkpoint has been published.
    directory = tmp_path / "failure"
    identity_path: Path | None = None

    def remember(recorder: LocalRunRecorder) -> None:
        nonlocal identity_path
        identity_path = recorder.run_dir

    def fail_at_60(batch: HostChunkBatch) -> None:
        if batch.end_step_exclusive == 60:
            raise RuntimeError("forced failure after checkpoint 40")

    with pytest.raises(RuntimeError, match="forced failure"):
        run_recorded_simulation(
            config=config,
            plan=RunPlan(100, 10),
            output_root=directory,
            capture_policy=ZarrCapturePolicy(checkpoint_interval=40),
            consumer=fail_at_60,
            on_created=remember,
        )
    assert identity_path is not None
    failed_manifest = load_run_manifest(identity_path / "manifest.json")
    assert failed_manifest.status is RunStatus.FAILED
    assert failed_manifest.error is not None
    assert (
        failed_manifest.error.last_confirmed_state_step,
        failed_manifest.error.last_committed_telemetry_step,
        failed_manifest.error.last_valid_checkpoint_step,
    ) == (60, 60, 40)

    stopped = StopController()

    class StopOnTelemetry:
        def created(self, _recorder: LocalRunRecorder) -> None:
            pass

        def consume(self, _recorder: LocalRunRecorder, telemetry: ChunkTelemetryDTO) -> None:
            if telemetry.end_step_exclusive == 70:
                stopped.request_stop()

        def finished(self, _recorder: LocalRunRecorder) -> None:
            pass

    intermediate = resume(identity_path, stop_controller=stopped, tracking=StopOnTelemetry())
    assert isinstance(intermediate, RecordedRunResult)
    assert intermediate.manifest.status is RunStatus.STOPPED
    assert intermediate.summary.final_step == 70
    final = resume(intermediate.run_dir, chunk_steps=4)
    assert isinstance(final, RecordedRunResult)
    chain = read_provenance(final.run_dir, final.manifest)
    assert chain and chain.recovery_depth == 2 and chain.root_run_id == failed_manifest.run_id
    assert chain.source_run_id == intermediate.manifest.run_id
    assert chain.source_checkpoint_step == 70
    assert_pytree_equivalent(final.result.runtime.state, expected.result.runtime.state)
    assert_pytree_equivalent(final.result.runtime.accumulator, expected.result.runtime.accumulator)
    for dataset in ("metrics", "births", "deaths", "genealogy"):
        assert (
            scan_dataset(final.run_dir, dataset)
            .collect()
            .drop("run_id")
            .equals(scan_dataset(expected.run_dir, dataset).collect().drop("run_id"))
        )
    assert load_run_manifest(identity_path / "manifest.json") == failed_manifest

    birth_part = next(interrupted.run_dir.glob("artifacts/parquet/births/part-00000000.parquet"))
    birth_part.unlink()
    report = inspect_run(interrupted.run_dir)
    assert report.latest_valid_checkpoint_step == 20
    assert report.latest_recoverable_checkpoint_step is None
