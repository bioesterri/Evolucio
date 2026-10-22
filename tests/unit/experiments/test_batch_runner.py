# ruff: noqa: ANN001, ANN003, ANN201

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from evolucio.experiments import (
    BatchExecutionError,
    BatchPlan,
    BatchRunStatus,
    ExperimentCase,
    SeedSet,
    run_batch,
    seeds_for_case,
)
from evolucio.runtime import RunMode


def fake_outcome(**kwargs):
    plan = kwargs["plan"]
    return SimpleNamespace(
        runtime=SimpleNamespace(
            steps_executed=plan.total_steps,
            final_step=plan.total_steps,
            state=SimpleNamespace(
                population=SimpleNamespace(
                    alive=np.array([True, False, True]),
                    lineage_id=np.array([1, -1, 1]),
                )
            ),
            accumulator=SimpleNamespace(births_total=np.array(2), deaths_total=np.array(1)),
        )
    )


@pytest.mark.parametrize("mode", list(RunMode))
def test_fresh_carries_factory_and_dispatch(config_path, monkeypatch, mode):
    init, empty = Mock(side_effect=lambda *a, **k: object()), Mock(side_effect=object)
    runtime = Mock(side_effect=fake_outcome)
    factory = Mock(side_effect=lambda spec: Mock())
    monkeypatch.setattr("evolucio.runtime.initialization.initialize_simulation", init)
    monkeypatch.setattr("evolucio.core.metrics.create_empty_metrics_accumulator", empty)
    monkeypatch.setattr("evolucio.experiments.batch.run_with_mode", runtime)
    plan = BatchPlan(
        (ExperimentCase("A", config_path), ExperimentCase("B", config_path)),
        SeedSet((101, 202, 303)),
        4,
        2,
        mode,
    )
    observed = []
    result = run_batch(plan=plan, consumer_factory=factory, on_result=observed.append)
    assert result.completed_run_count == result.requested_run_count == 6
    assert result.failed_run_count == 0
    assert tuple(observed) == result.runs
    assert seeds_for_case(result, "A") == seeds_for_case(result, "B") == (101, 202, 303)
    calls = runtime.call_args_list
    assert len({id(call.kwargs["state"]) for call in calls}) == 6
    assert len({id(call.kwargs["accumulator"]) for call in calls}) == 6
    assert [call.kwargs["seed"] for call in init.call_args_list] == [101, 202, 303] * 2
    for call, factory_call in zip(calls, factory.call_args_list, strict=True):
        assert call.kwargs["mode"] == mode
        assert factory_call.args[0].seed in (101, 202, 303)
    assert result.runs[0].alive_count == 2
    assert result.runs[0].active_lineage_count == 1
    assert result.runs[0].births_total == 2


@pytest.mark.parametrize("phase", ["runtime", "initialization", "factory"])
def test_errors_continue_and_fail_fast(config_path, monkeypatch, phase):
    runtime = Mock(side_effect=fake_outcome)
    initializer = Mock(side_effect=lambda *a, **k: object())
    factory = Mock(return_value=None)
    monkeypatch.setattr("evolucio.experiments.batch.run_with_mode", runtime)
    monkeypatch.setattr("evolucio.runtime.initialization.initialize_simulation", initializer)
    failing = {"runtime": runtime, "initialization": initializer, "factory": factory}[phase]
    success = fake_outcome(plan=SimpleNamespace(total_steps=4)) if phase == "runtime" else None
    failing.side_effect = [RuntimeError("technical failure"), success, success]
    plan = BatchPlan((ExperimentCase("A", config_path),), SeedSet((11, 22, 33)), 4, 2)
    result = run_batch(plan=plan, consumer_factory=factory)
    assert (result.completed_run_count, result.failed_run_count) == (2, 1)
    failure = result.runs[0]
    assert (failure.case_id, failure.seed, failure.status) == ("A", 11, BatchRunStatus.FAILED)
    assert failure.error_type == "RuntimeError" and failure.error_message == "technical failure"
    assert failure.steps_executed is None and failure.final_step is None
    failing.reset_mock(side_effect=True)
    failing.side_effect = RuntimeError("stop")
    runtime.reset_mock()
    initializer.reset_mock()
    observer = Mock()
    with pytest.raises(BatchExecutionError) as caught:
        run_batch(
            plan=replace(plan, continue_on_error=False),
            consumer_factory=factory,
            on_result=observer,
        )
    assert len(caught.value.runs) == 1
    assert caught.value.requested_run_count == 3
    observer.assert_called_once()
    assert initializer.call_count == 1


def test_no_consumer_and_observer_errors(config_path, monkeypatch):
    runtime = Mock(side_effect=fake_outcome)
    monkeypatch.setattr("evolucio.experiments.batch.run_with_mode", runtime)
    plan = BatchPlan((ExperimentCase("A", config_path),), SeedSet((11,)), 4, 2)
    run_batch(plan=plan)
    assert runtime.call_args.kwargs["consumer"] is None
    with pytest.raises(RuntimeError, match="observer"):
        run_batch(plan=plan, on_result=Mock(side_effect=RuntimeError("observer")))
