# ruff: noqa: ANN001, ANN201, ANN202

import weakref
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from evolucio.runtime import HostChunkBatch, RunPlan, RunRuntimeResult, run_simulation, runner


@pytest.fixture
def kernel(monkeypatch):
    calls = []
    order = []
    carry = [SimpleNamespace(step=3), object()]
    original_state, original_accumulator = carry
    previous_outputs = []

    def run_chunk(*, state, accumulator, config, chunk_steps):
        assert state is carry[0]
        assert accumulator is carry[1]
        # The runtime must release outputs before dispatching the next chunk.
        assert all(reference() is None for reference in previous_outputs)
        calls.append(chunk_steps)
        order.append("kernel")
        steps = np.arange(state.step, state.step + chunk_steps, dtype=np.int32)
        events = steps.copy()
        previous_outputs[:] = [weakref.ref(steps), weakref.ref(events)]
        carry[:] = [SimpleNamespace(step=state.step + chunk_steps), object()]
        return SimpleNamespace(
            state=carry[0],
            accumulator=carry[1],
            metrics=SimpleNamespace(step=steps),
            events=events,
        )

    def transfer(outputs):
        assert len(outputs) == 2
        assert hasattr(outputs[0], "step")
        assert isinstance(outputs[1], np.ndarray)
        order.append("transfer")
        return outputs

    monkeypatch.setattr(runner, "run_chunk", run_chunk)
    monkeypatch.setattr(runner.jax, "device_get", transfer)
    return original_state, original_accumulator, carry, calls, order


@pytest.mark.parametrize(
    ("total", "size", "lengths"), [(10, 4, [4, 4, 2]), (8, 4, [4, 4]), (2, 4, [2]), (1, 1, [1])]
)
@pytest.mark.parametrize("with_consumer", [True, False])
def test_exact_streaming_plan(kernel, total, size, lengths, with_consumer):
    state, accumulator, carry, calls, order = kernel
    intervals = []

    def consume(batch):
        order.append("consumer")
        assert isinstance(batch, HostChunkBatch)
        assert isinstance(batch.metrics.step, np.ndarray)
        assert isinstance(batch.events, np.ndarray)
        np.testing.assert_array_equal(batch.events, batch.metrics.step)
        assert batch.metrics.step.tolist() == list(
            range(batch.start_step, batch.end_step_exclusive)
        )
        intervals.append((batch.chunk_index, batch.start_step, batch.end_step_exclusive))

    result = run_simulation(
        state=state,
        accumulator=accumulator,
        config=object(),
        plan=RunPlan(total_steps=total, chunk_steps=size),
        consumer=consume if with_consumer else None,
    )
    assert calls == lengths
    assert result.state is carry[0]
    assert result.accumulator is carry[1]
    assert (result.initial_step, result.final_step) == (3, 3 + total)
    assert (result.steps_executed, result.chunks_executed) == (total, len(lengths))
    assert order == (
        ["kernel", "transfer", "consumer"] if with_consumer else ["kernel", "transfer"]
    ) * len(lengths)
    if with_consumer:
        assert intervals[0] == (0, 3, 3 + lengths[0])
        assert intervals[-1][2] == 3 + total


def test_consumer_exception_stops_future_chunks(kernel):
    state, accumulator, _, calls, order = kernel
    error = RuntimeError("sink failed")

    def consume(batch):
        raise error

    with pytest.raises(RuntimeError) as caught:
        run_simulation(
            state=state,
            accumulator=accumulator,
            config=object(),
            plan=RunPlan(10, 4),
            consumer=consume,
        )
    assert caught.value is error
    assert calls == [4]
    assert order == ["kernel", "transfer"]


def test_simulation_exception_propagates(monkeypatch):
    error = RuntimeError("kernel failed")

    def fail(**kwargs: object):
        raise error

    monkeypatch.setattr(runner, "run_chunk", fail)
    with pytest.raises(RuntimeError) as caught:
        run_simulation(
            state=SimpleNamespace(step=0),
            accumulator=object(),
            config=object(),
            plan=RunPlan(10, 4),
        )
    assert caught.value is error


@pytest.mark.parametrize("steps", [[3, 5, 5, 6], [3, 4, 5], [4, 5, 6, 7], [[3, 4, 5, 6]]])
def test_inconsistent_metric_intervals_are_rejected(monkeypatch, steps):
    monkeypatch.setattr(
        runner,
        "run_chunk",
        lambda **kwargs: SimpleNamespace(
            state=SimpleNamespace(step=7),
            accumulator=object(),
            metrics=SimpleNamespace(step=np.asarray(steps)),
            events=object(),
        ),
    )
    monkeypatch.setattr(runner.jax, "device_get", lambda outputs: outputs)
    with pytest.raises(ValueError, match=r"metrics\.step"):
        run_simulation(
            state=SimpleNamespace(step=3),
            accumulator=object(),
            config=object(),
            plan=RunPlan(4, 4),
            consumer=lambda batch: pytest.fail("invalid batch"),
        )


def test_final_state_step_is_checked(monkeypatch):
    monkeypatch.setattr(
        runner,
        "run_chunk",
        lambda **kwargs: SimpleNamespace(
            state=SimpleNamespace(step=3),
            accumulator=object(),
            metrics=SimpleNamespace(step=np.asarray([3])),
            events=object(),
        ),
    )
    monkeypatch.setattr(runner.jax, "device_get", lambda outputs: outputs)
    with pytest.raises(ValueError, match=r"final state\.step"):
        run_simulation(
            state=SimpleNamespace(step=3), accumulator=object(), config=object(), plan=RunPlan(1, 1)
        )


def test_contracts_do_not_expose_state_or_history():
    assert {field.name for field in fields(HostChunkBatch)} == {
        "chunk_index",
        "start_step",
        "end_step_exclusive",
        "metrics",
        "events",
    }
    assert {field.name for field in fields(RunRuntimeResult)} == {
        "state",
        "accumulator",
        "initial_step",
        "final_step",
        "steps_executed",
        "chunks_executed",
    }
