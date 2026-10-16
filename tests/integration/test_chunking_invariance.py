# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import jax
import jax.numpy as jnp
import pytest

from evolucio.core.metrics import create_empty_metrics_accumulator
from evolucio.core.simulation import run_chunk
from tests.regression.helpers import assert_pytree_equivalent, canonical_case


@pytest.mark.parametrize("partition", [(8, 8, 8, 8), (7, 13, 12)])
def test_thirty_two_steps_are_independent_of_chunk_partition(partition: tuple[int, ...]) -> None:
    whole_state, config, _ = canonical_case()
    whole = run_chunk(
        state=whole_state,
        accumulator=create_empty_metrics_accumulator(),
        config=config,
        chunk_steps=32,
    )
    state, _, _ = canonical_case()
    accumulator = create_empty_metrics_accumulator()
    metrics = []
    events = []
    for length in partition:
        part = run_chunk(state=state, accumulator=accumulator, config=config, chunk_steps=length)
        state, accumulator = part.state, part.accumulator
        metrics.append(part.metrics)
        events.append(part.events)
    concatenated_metrics = jax.tree.map(lambda *values: jnp.concatenate(values), *metrics)
    concatenated_events = jax.tree.map(lambda *values: jnp.concatenate(values), *events)
    assert_pytree_equivalent(whole.state, state)
    assert_pytree_equivalent(whole.accumulator, accumulator)
    assert_pytree_equivalent(whole.metrics, concatenated_metrics)
    assert_pytree_equivalent(whole.events, concatenated_events)
