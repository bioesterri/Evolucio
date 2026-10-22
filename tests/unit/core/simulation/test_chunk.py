# ruff: noqa: ANN001, ANN201, ANN202

import jax
import jax.numpy as jnp
import pytest

from evolucio.core.codes import ActionCode
from evolucio.core.metrics import create_empty_metrics_accumulator
from evolucio.core.simulation import run_chunk, run_chunk_impl, simulation_step


def assert_tree_equal(left, right) -> None:
    def equal(left_leaf, right_leaf):
        if jnp.issubdtype(left_leaf.dtype, jnp.inexact):
            return jnp.allclose(left_leaf, right_leaf, rtol=1e-6, atol=1e-6)
        return jnp.array_equal(left_leaf, right_leaf)

    assert jax.tree.all(jax.tree.map(equal, left, right))


def concatenate(first, second):
    return jax.tree.map(lambda left, right: jnp.concatenate((left, right)), first, second)


@pytest.mark.parametrize("chunk_steps", [0, -1])
def test_non_positive_chunk_length_is_rejected(step_case, chunk_steps: int) -> None:
    state, config = step_case(alive=False)
    with pytest.raises(ValueError, match="strictly positive"):
        run_chunk_impl(
            state=state,
            accumulator=create_empty_metrics_accumulator(),
            config=config,
            chunk_steps=chunk_steps,
        )


def test_boolean_chunk_length_is_rejected(step_case) -> None:
    state, config = step_case(alive=False)
    with pytest.raises(TypeError, match="Python int"):
        run_chunk_impl(
            state=state,
            accumulator=create_empty_metrics_accumulator(),
            config=config,
            chunk_steps=True,
        )


def test_one_step_chunk_is_exactly_simulation_step(step_case) -> None:
    state, config = step_case(action=ActionCode.STAY)
    accumulator = create_empty_metrics_accumulator()
    step = simulation_step(state=state, accumulator=accumulator, config=config)
    chunk = run_chunk_impl(state=state, accumulator=accumulator, config=config, chunk_steps=1)
    assert_tree_equal(step.state, chunk.state)
    assert_tree_equal(step.accumulator, chunk.accumulator)
    assert_tree_equal(step.metrics, jax.tree.map(lambda value: value[0], chunk.metrics))
    assert_tree_equal(step.events, jax.tree.map(lambda value: value[0], chunk.events))


def test_compiled_chunk_matches_individual_steps_and_has_fixed_outputs(step_case) -> None:
    state, config = step_case(action=ActionCode.STAY)
    accumulator = create_empty_metrics_accumulator()
    reference_metrics = []
    reference_events = []
    reference_state = state
    reference_accumulator = accumulator
    for _ in range(3):
        result = simulation_step(
            state=reference_state, accumulator=reference_accumulator, config=config
        )
        reference_state, reference_accumulator = result.state, result.accumulator
        reference_metrics.append(result.metrics)
        reference_events.append(result.events)
    stacked_metrics = jax.tree.map(lambda *values: jnp.stack(values), *reference_metrics)
    stacked_events = jax.tree.map(lambda *values: jnp.stack(values), *reference_events)

    eager = run_chunk_impl(state=state, accumulator=accumulator, config=config, chunk_steps=3)
    compiled = run_chunk(state=state, accumulator=accumulator, config=config, chunk_steps=3)
    compiled_again = run_chunk(state=state, accumulator=accumulator, config=config, chunk_steps=3)
    for actual in (eager, compiled, compiled_again):
        assert_tree_equal(reference_state, actual.state)
        assert_tree_equal(reference_accumulator, actual.accumulator)
        assert_tree_equal(stacked_metrics, actual.metrics)
        assert_tree_equal(stacked_events, actual.events)
    assert compiled.state.step.shape == ()
    assert compiled.state.population.alive.shape == (4,)
    assert compiled.metrics.alive_count.shape == (3,)
    assert compiled.metrics.deaths_by_cause.shape == (3, 6)
    assert compiled.events.births.born.shape == (3, 4)
    assert compiled.events.births.birth_position.shape == (3, 4, 2)
    assert compiled.events.deaths.records.died.shape == (3, 4)


def test_extinction_does_not_stop_remaining_steps(step_case) -> None:
    state, config = step_case(
        action=ActionCode.STAY,
        energy=0.15,
        energy_changes={"basal_cost": 0.1},
    )
    result = run_chunk(
        state=state,
        accumulator=create_empty_metrics_accumulator(),
        config=config,
        chunk_steps=4,
    )
    assert result.metrics.alive_count.tolist() == [1, 0, 0, 0]
    assert result.metrics.step.tolist() == [3, 4, 5, 6]
    assert result.state.step.item() == 7
    assert result.events.deaths.records.died.shape == (4, 4)
    assert result.accumulator.steps_accumulated.item() == 4


def test_empty_population_still_executes_every_step_deterministically(step_case) -> None:
    state, config = step_case(alive=False)
    accumulator = create_empty_metrics_accumulator()
    first = run_chunk(state=state, accumulator=accumulator, config=config, chunk_steps=3)
    second = run_chunk(state=state, accumulator=accumulator, config=config, chunk_steps=3)
    assert first.metrics.alive_count.tolist() == [0, 0, 0]
    assert first.state.step.item() == 6
    assert_tree_equal(first, second)


def test_movement_and_feeding_continue_across_steps(step_case) -> None:
    moving_state, moving_config = step_case(action=ActionCode.MOVE_EAST)
    moved = run_chunk_impl(
        state=moving_state,
        accumulator=create_empty_metrics_accumulator(),
        config=moving_config,
        chunk_steps=2,
    )
    assert moved.state.population.position[0].tolist() == [4, 2]
    assert moved.metrics.movement_success_count.tolist() == [1, 1]

    feeding_state, feeding_config = step_case(action=ActionCode.EAT, energy=20)
    fed = run_chunk_impl(
        state=feeding_state,
        accumulator=create_empty_metrics_accumulator(),
        config=feeding_config,
        chunk_steps=2,
    )
    assert bool(jnp.all(fed.metrics.resource_consumed > 0))
    assert bool(jnp.all(fed.metrics.feeding_success_count == 1))


def test_reproduction_mutation_and_genealogy_are_stacked(step_case) -> None:
    state, config = step_case(
        action=ActionCode.REPRODUCE,
        energy=60,
        energy_changes={"basal_cost": 0.01},
        evolution_changes={
            "weight_mutation_rate": 1.0,
            "weight_mutation_sigma": 0.1,
            "bias_mutation_rate": 1.0,
            "bias_mutation_sigma": 0.1,
        },
    )
    result = run_chunk_impl(
        state=state,
        accumulator=create_empty_metrics_accumulator(),
        config=config,
        chunk_steps=2,
    )
    first_newborn = result.events.births.born[0]
    child_slot = int(jnp.argmax(first_newborn))
    assert result.metrics.birth_count[0].item() == 1
    assert result.events.births.parent_agent_id[0, child_slot].item() == 0
    assert result.events.births.lineage_id[0, child_slot].item() == 0
    assert result.events.births.generation[0, child_slot].item() == 1
    assert result.events.births.mutation_effective_count[0, child_slot].item() > 0
    assert result.state.population.age[child_slot].item() == 1
    assert result.state.ids.next_agent_id.item() >= 2


def test_chunking_invariance_whole_equals_composition(step_case) -> None:
    state, config = step_case(action=ActionCode.STAY)
    accumulator = create_empty_metrics_accumulator()
    whole = run_chunk(state=state, accumulator=accumulator, config=config, chunk_steps=8)
    first = run_chunk(state=state, accumulator=accumulator, config=config, chunk_steps=3)
    second = run_chunk(
        state=first.state,
        accumulator=first.accumulator,
        config=config,
        chunk_steps=5,
    )
    assert_tree_equal(whole.state, second.state)
    assert_tree_equal(whole.accumulator, second.accumulator)
    assert_tree_equal(whole.metrics, concatenate(first.metrics, second.metrics))
    assert_tree_equal(whole.events, concatenate(first.events, second.events))
