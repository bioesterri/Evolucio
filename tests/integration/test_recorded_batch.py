# ruff: noqa: ANN001, ANN201, ANN202

import pytest

from evolucio.experiments import BatchExecutionError, BatchPlan, SeedSet, run_batch
from evolucio.persistence.local import RunStatus, load_run_manifest, verify_local_run
from tests.experiment_helpers import write_cases


@pytest.mark.parametrize("fail_fast", [False, True])
def test_batch_separate_records_failure_and_continuation(tmp_path, monkeypatch, fail_fast):
    from evolucio.runtime import initialization

    original = initialization.initialize_simulation

    def initialize(config, *, seed):
        if seed == 11:
            raise RuntimeError("failed seed 11")
        return original(config, seed=seed)

    monkeypatch.setattr(initialization, "initialize_simulation", initialize)
    plan = BatchPlan(
        write_cases(tmp_path), SeedSet((11, 22)), 4, 2, continue_on_error=not fail_fast
    )
    root = tmp_path / "runs"
    if fail_fast:
        with pytest.raises(BatchExecutionError) as caught:
            run_batch(plan=plan, output_root=root)
        runs = caught.value.runs
        assert len(runs) == 1
    else:
        result = run_batch(plan=plan, output_root=root)
        assert result.completed_run_count == result.failed_run_count == 2
        runs = result.runs
    directories = list(root.iterdir())
    assert len(directories) == len(runs)
    assert len({r.run_id for r in runs}) == len(runs)
    manifests = {
        load_run_manifest(d / "manifest.json").run_id: load_run_manifest(d / "manifest.json")
        for d in directories
    }
    for run in runs:
        manifest = manifests[run.run_id]
        assert (manifest.case_id, manifest.seed, manifest.batch_ordinal) == (
            run.case_id,
            run.seed,
            run.ordinal,
        )
        assert manifest.config_hash == run.config_hash
        assert manifest.status == (RunStatus.FAILED if run.seed == 11 else RunStatus.COMPLETED)
        assert verify_local_run(root / run.run_id).ok


def test_bad_batch_never_creates_record(tmp_path):
    cases = write_cases(tmp_path)
    cases[1].config_path.write_text("{}")
    root = tmp_path / "runs"
    with pytest.raises(ValueError):
        run_batch(plan=BatchPlan(cases, SeedSet((11,)), 2, 1), output_root=root)
    assert not root.exists()


def test_batch_stop_does_not_start_pending_runs(tmp_path):
    from evolucio.experiments import BatchRunStatus
    from evolucio.runtime.stop_control import StopController

    controller = StopController()
    root = tmp_path / "runs"
    plan = BatchPlan(write_cases(tmp_path), SeedSet((11, 22)), 4, 2)

    def make_consumer(spec):
        def consume(_batch):
            if spec.ordinal == 2:
                controller.request_stop()

        return consume

    result = run_batch(
        plan=plan, output_root=root, consumer_factory=make_consumer, stop_controller=controller
    )
    assert [run.status for run in result.runs] == [
        BatchRunStatus.COMPLETED,
        BatchRunStatus.STOPPED,
        BatchRunStatus.NOT_STARTED,
        BatchRunStatus.NOT_STARTED,
    ]
    assert result.failed_run_count == 0
    assert result.stopped_run_count == 1
    assert result.not_started_run_count == 2
    assert len(list(root.iterdir())) == 2
