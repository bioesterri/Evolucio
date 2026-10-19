# ruff: noqa: ANN001, ANN201

from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import pytest

from evolucio.runtime import RunMode, RunPlan, RunRuntimeResult, run_with_mode


@pytest.mark.parametrize("steps,chunk,lengths", [(10, 4, [4, 2]), (8, 4, [4]), (2, 4, [2])])
def test_warmup_and_timer_order(monkeypatch, steps, chunk, lengths):
    from evolucio.runtime import benchmark

    state = SimpleNamespace(step=jnp.asarray(3))
    accumulator, config = object(), object()
    result = RunRuntimeResult(state, accumulator, 3, 3 + steps, steps, (steps + chunk - 1) // chunk)
    order, inputs = [], []

    def warmup(**kwargs: object) -> str:
        order.append(("warm", kwargs["chunk_steps"]))
        inputs.append((kwargs["state"], kwargs["accumulator"]))
        return "warm"

    def sync(value) -> None:
        order.append(("sync", value))

    def runtime(**kwargs: object) -> RunRuntimeResult:
        order.append(("run", kwargs["consumer"]))
        assert kwargs["state"] is state
        assert kwargs["accumulator"] is accumulator
        return result

    def clock() -> float:
        order.append(("clock", None))
        return 10.0 if sum(name == "clock" for name, _ in order) == 1 else 12.0

    monkeypatch.setattr(benchmark, "run_chunk", warmup)
    monkeypatch.setattr(benchmark.jax, "block_until_ready", sync)
    monkeypatch.setattr(benchmark, "run_simulation", runtime)
    monkeypatch.setattr(benchmark, "perf_counter", clock)
    consumer = Mock()
    outcome = run_with_mode(
        mode=RunMode.BENCHMARK,
        state=state,
        accumulator=accumulator,
        config=config,
        plan=RunPlan(steps, chunk),
        consumer=consumer,
    )
    assert [value for name, value in order if name == "warm"] == lengths
    assert all(s is state and a is accumulator for s, a in inputs)
    assert [name for name, _ in order][-4:] == ["clock", "run", "sync", "clock"]
    assert order[-3][1] is None
    assert outcome.benchmark.elapsed_seconds == 2.0
    assert outcome.benchmark.steps_per_second == steps / 2
    assert outcome.benchmark.final_step == 3 + steps
    consumer.assert_not_called()


def test_invalid_clock_is_not_reported_as_performance(monkeypatch):
    from evolucio.runtime import benchmark

    state = SimpleNamespace(step=jnp.asarray(0))
    monkeypatch.setattr(benchmark, "run_chunk", Mock())
    monkeypatch.setattr(benchmark.jax, "block_until_ready", Mock())
    monkeypatch.setattr(
        benchmark,
        "run_simulation",
        Mock(return_value=SimpleNamespace(state=state, accumulator=None)),
    )
    monkeypatch.setattr(benchmark, "perf_counter", Mock(return_value=1.0))
    with pytest.raises(RuntimeError, match="strictly positive"):
        run_with_mode(
            mode=RunMode.BENCHMARK, state=state, accumulator=None, config=None, plan=RunPlan(1, 1)
        )
