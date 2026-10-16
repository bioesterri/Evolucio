# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.diagnostics import (
    check_final_state_invariants,
    check_step_transition_invariants,
    check_structural_invariants,
)
from evolucio.core.metrics import create_empty_metrics_accumulator
from evolucio.core.simulation import (
    run_chunk,
    run_chunk_impl,
    simulation_step,
    simulation_step_checked,
)
from tests.regression.helpers import (
    REGRESSION_STEPS,
    assert_pytree_equivalent,
    canonical_case,
    first_divergent_step,
)


def test_simulation_step_eager_equals_jit_and_invariants_hold() -> None:
    state, config, _ = canonical_case()
    accumulator = create_empty_metrics_accumulator()
    eager = simulation_step(state=state, accumulator=accumulator, config=config)
    compiled = eqx.filter_jit(simulation_step)(state=state, accumulator=accumulator, config=config)
    assert_pytree_equivalent(eager, compiled)
    initial = check_structural_invariants(
        state=state,
        resource_capacity=config.world.resource_capacity,
        width=config.world.width,
        height=config.world.height,
    )
    final = check_final_state_invariants(
        state=eager.state,
        resource_capacity=config.world.resource_capacity,
        death_energy_threshold=config.energy.death_threshold,
        maximum_age=config.evolution.max_age,
        width=config.world.width,
        height=config.world.height,
    )
    transition = check_step_transition_invariants(
        before=state,
        after=eager.state,
        events=eager.events,
        metrics=eager.metrics,
        accumulator=eager.accumulator,
    )
    assert bool(initial.ok) and bool(final.ok) and bool(transition.ok)


def test_independent_same_seed_runs_and_compiled_chunk_are_equivalent() -> None:
    state_a, config_a, _ = canonical_case()
    state_b, config_b, _ = canonical_case()
    accumulator_a = create_empty_metrics_accumulator()
    accumulator_b = create_empty_metrics_accumulator()
    eager = run_chunk_impl(
        state=state_a,
        accumulator=accumulator_a,
        config=config_a,
        chunk_steps=REGRESSION_STEPS,
    )
    compiled = run_chunk(
        state=state_b,
        accumulator=accumulator_b,
        config=config_b,
        chunk_steps=REGRESSION_STEPS,
    )
    assert first_divergent_step(eager.metrics, compiled.metrics) is None
    assert_pytree_equivalent(eager, compiled)


def test_different_seed_changes_stochastic_results() -> None:
    first_state, first_config, _ = canonical_case(1729)
    second_state, second_config, _ = canonical_case(1730)
    first = run_chunk(
        state=first_state,
        accumulator=create_empty_metrics_accumulator(),
        config=first_config,
        chunk_steps=2,
    )
    second = run_chunk(
        state=second_state,
        accumulator=create_empty_metrics_accumulator(),
        config=second_config,
        chunk_steps=2,
    )
    assert not bool(jnp.array_equal(first.state.rng.key, second.state.rng.key))
    assert not bool(
        jnp.array_equal(first.state.genomes.layer1.weight, second.state.genomes.layer1.weight)
    )


def test_step_by_step_matches_chunk() -> None:
    state, config, _ = canonical_case()
    accumulator = create_empty_metrics_accumulator()
    metrics = []
    events = []
    for _ in range(REGRESSION_STEPS):
        result = simulation_step(state=state, accumulator=accumulator, config=config)
        state, accumulator = result.state, result.accumulator
        metrics.append(result.metrics)
        events.append(result.events)
    expected_metrics = jax.tree.map(lambda *leaves: jnp.stack(leaves), *metrics)
    expected_events = jax.tree.map(lambda *leaves: jnp.stack(leaves), *events)
    chunk_state, chunk_config, _ = canonical_case()
    chunk = run_chunk(
        state=chunk_state,
        accumulator=create_empty_metrics_accumulator(),
        config=chunk_config,
        chunk_steps=REGRESSION_STEPS,
    )
    assert_pytree_equivalent(state, chunk.state)
    assert_pytree_equivalent(accumulator, chunk.accumulator)
    assert_pytree_equivalent(expected_metrics, chunk.metrics)
    assert_pytree_equivalent(expected_events, chunk.events)


def test_every_canonical_transition_and_final_state_satisfies_invariants() -> None:
    state, config, _ = canonical_case()
    accumulator = create_empty_metrics_accumulator()
    for _ in range(REGRESSION_STEPS):
        checked = simulation_step_checked(state=state, accumulator=accumulator, config=config)
        assert bool(checked.initial_invariants.ok)
        assert bool(checked.final_invariants.ok)
        assert bool(checked.transition_invariants.ok)
        state, accumulator = checked.result.state, checked.result.accumulator
