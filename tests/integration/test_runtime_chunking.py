# ruff: noqa: ANN001, ANN201, ANN202

from dataclasses import fields

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evolucio.core.codes import ActionCode
from evolucio.core.metrics import create_empty_metrics_accumulator
from evolucio.core.simulation import run_chunk
from evolucio.runtime import HostChunkBatch, RunPlan, run_simulation
from tests.regression.helpers import assert_pytree_equivalent, canonical_case


def concatenate(batches, name):
    return jax.tree.map(
        lambda *values: np.concatenate(values), *(getattr(batch, name) for batch in batches)
    )


def assert_host_batch(batch):
    assert isinstance(batch, HostChunkBatch)
    assert {field.name for field in fields(batch)} == {
        "chunk_index",
        "start_step",
        "end_step_exclusive",
        "metrics",
        "events",
    }
    assert all(
        isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves((batch.metrics, batch.events))
    )


@pytest.mark.parametrize("size", [1, 2, 4, 5, 7, 10, 16])
def test_twenty_steps_are_neutral_and_repeatable(size):
    state, config, _ = canonical_case()
    accumulator = create_empty_metrics_accumulator()
    whole = run_chunk(state=state, accumulator=accumulator, config=config, chunk_steps=20)
    batches = []

    def consume(batch):
        assert_host_batch(batch)
        batches.append(batch)

    result = run_simulation(
        state=state,
        accumulator=accumulator,
        config=config,
        plan=RunPlan(20, size),
        consumer=consume,
    )
    repeated = run_simulation(
        state=state, accumulator=accumulator, config=config, plan=RunPlan(20, size)
    )
    for actual in (result, repeated):
        assert actual.final_step == int(actual.state.step) == 20
        assert actual.steps_executed == 20
        assert_pytree_equivalent(actual.state, whole.state)
        assert_pytree_equivalent(actual.state.rng, whole.state.rng)
        assert_pytree_equivalent(actual.state.ids, whole.state.ids)
        assert_pytree_equivalent(actual.accumulator, whole.accumulator)
    metrics = concatenate(batches, "metrics")
    events = concatenate(batches, "events")
    assert_pytree_equivalent(metrics, jax.device_get(whole.metrics))
    assert_pytree_equivalent(events, jax.device_get(whole.events))
    assert metrics.step.tolist() == list(range(20))
    assert int(events.births.born.sum()) == int(result.accumulator.births_total) > 0
    assert int(events.deaths.records.died.sum()) == int(result.accumulator.deaths_total) > 0


def test_ten_equals_four_four_two_from_nonzero_step_and_accumulator():
    state, config, _ = canonical_case()
    warm = run_chunk(
        state=state, accumulator=create_empty_metrics_accumulator(), config=config, chunk_steps=3
    )
    whole = run_chunk(state=warm.state, accumulator=warm.accumulator, config=config, chunk_steps=10)
    batches = []
    result = run_simulation(
        state=warm.state,
        accumulator=warm.accumulator,
        config=config,
        plan=RunPlan(10, 4),
        consumer=batches.append,
    )
    assert [(b.chunk_index, b.start_step, b.end_step_exclusive) for b in batches] == [
        (0, 3, 7),
        (1, 7, 11),
        (2, 11, 13),
    ]
    assert [b.metrics.step.shape[0] for b in batches] == [4, 4, 2]
    assert concatenate(batches, "metrics").step.tolist() == list(range(3, 13))
    assert result.final_step == 13
    assert result.chunks_executed == 3
    assert int(result.accumulator.steps_accumulated) == 13
    assert_pytree_equivalent(result.state, whole.state)
    assert_pytree_equivalent(result.accumulator, whole.accumulator)
    assert_pytree_equivalent(concatenate(batches, "events"), jax.device_get(whole.events))


@pytest.mark.parametrize("empty", [True, False])
def test_empty_population_and_extinction_continue_to_horizon(empty):
    state, config, _ = canonical_case()
    alive = jnp.zeros_like(state.population.alive) if empty else state.population.alive
    state = eqx.tree_at(
        lambda s: (s.population.alive, s.population.energy),
        state,
        (alive, jnp.full_like(state.population.energy, 0.15)),
    )
    biases = jnp.zeros_like(state.genomes.layer2.bias).at[:, int(ActionCode.STAY)].set(2)
    state = eqx.tree_at(lambda s: s.genomes.layer2.bias, state, biases)
    batches = []
    result = run_simulation(
        state=state,
        accumulator=create_empty_metrics_accumulator(),
        config=config,
        plan=RunPlan(10, 4),
        consumer=batches.append,
    )
    assert result.final_step == 10
    assert result.chunks_executed == 3
    metrics = concatenate(batches, "metrics")
    assert metrics.step.tolist() == list(range(10))
    assert metrics.alive_count[-1] == 0
    assert np.all(metrics.alive_count[2:] == 0)
    assert int(result.accumulator.steps_accumulated) == 10
    events = concatenate(batches, "events")
    assert int(events.deaths.records.died.sum()) == (0 if empty else 3)
