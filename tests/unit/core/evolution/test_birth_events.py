# ruff: noqa: ANN001, ANN201, ANN202

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.dtypes import COUNT_DTYPE, REAL_DTYPE
from evolucio.core.evolution import (
    BirthEventBatch,
    build_birth_events,
    mutate_newborn_genomes,
    resolve_asexual_reproduction,
)
from evolucio.core.state import PopulationState


def _resolve_and_mutate(case, *, rate: float = 1.0):
    reproduction = resolve_asexual_reproduction(**case)
    mutation = mutate_newborn_genomes(
        genomes=reproduction.genomes,
        population=reproduction.population,
        newborn_mask=reproduction.newborn_mask,
        genome_mutation_key=jax.random.key(40),
        weight_mutation_rate=jnp.asarray(rate, dtype=REAL_DTYPE),
        weight_mutation_sigma=jnp.asarray(0.1, dtype=REAL_DTYPE),
        weight_abs_limit=jnp.asarray(100.0, dtype=REAL_DTYPE),
        bias_mutation_rate=jnp.asarray(rate, dtype=REAL_DTYPE),
        bias_mutation_sigma=jnp.asarray(0.1, dtype=REAL_DTYPE),
        bias_abs_limit=jnp.asarray(100.0, dtype=REAL_DTYPE),
    )
    events = build_birth_events(
        population_before_reproduction=case["population"],
        reproduction=reproduction,
        mutation=mutation,
        step=case["step"],
    )
    return reproduction, mutation, events


def test_integrated_birth_event_audits_identity_genome_energy_and_mutation(reproduction_case):
    reproduction, mutation, events = _resolve_and_mutate(reproduction_case)
    child = int(jnp.flatnonzero(events.born, size=1)[0])

    assert isinstance(events, BirthEventBatch)
    assert jnp.array_equal(events.born, reproduction.newborn_mask)
    assert int(jnp.sum(events.born)) == int(reproduction.birth_count) == 1
    assert int(events.child_agent_id[child]) == 20
    assert int(events.parent_agent_id[child]) == 7
    assert int(events.lineage_id[child]) == 4
    assert int(events.generation[child]) == 3
    assert int(events.birth_step[child]) == 12
    assert int(events.child_genome_id[child]) == 30
    assert int(events.parent_genome_id[child]) == 9
    assert jnp.array_equal(events.birth_position[child], reproduction.birth_positions[child])
    assert float(events.child_initial_energy[child]) == 3
    assert float(events.parent_energy_before[child]) == 10
    assert float(events.parent_energy_after[child]) == 6
    assert int(events.mutation_selected_count[child]) == int(
        mutation.selected_parameter_count[child]
    )
    assert int(events.mutation_effective_count[child]) == int(
        mutation.effective_parameter_count[child]
    )
    assert int(events.mutation_selected_weight_count[child]) == int(
        mutation.selected_weight_count[child]
    )
    assert int(events.mutation_selected_bias_count[child]) == int(
        mutation.selected_bias_count[child]
    )
    assert float(events.mutation_sum_abs_delta[child]) == float(mutation.sum_abs_delta[child])
    assert float(events.mutation_max_abs_delta[child]) == float(mutation.max_abs_delta[child])
    assert not {"weight", "bias", "event_id"} & {field.name for field in dataclasses.fields(events)}


def test_nonbirth_rows_are_canonical_and_shapes_and_dtypes_are_fixed(reproduction_case):
    _, _, events = _resolve_and_mutate(reproduction_case)
    inactive = ~events.born

    assert events.born.shape == (3,)
    assert events.birth_position.shape == (3, 2)
    assert events.born.dtype == jnp.bool_
    assert events.birth_step.dtype == events.child_agent_id.dtype == jnp.int32
    assert events.child_initial_energy.dtype == events.mutation_sum_abs_delta.dtype == jnp.float32
    for values in (
        events.child_agent_id,
        events.parent_agent_id,
        events.lineage_id,
        events.child_genome_id,
        events.parent_genome_id,
    ):
        assert jnp.all(values[inactive] == -1)
    for values in (
        events.birth_step,
        events.generation,
        events.child_initial_energy,
        events.parent_energy_before,
        events.parent_energy_after,
        events.mutation_selected_count,
        events.mutation_effective_count,
        events.mutation_selected_weight_count,
        events.mutation_selected_bias_count,
        events.mutation_sum_abs_delta,
        events.mutation_max_abs_delta,
    ):
        assert jnp.all(values[inactive] == 0)
    assert jnp.all(events.birth_position[inactive] == -1)


def test_failed_reproduction_has_no_event(reproduction_case):
    case = dict(reproduction_case)
    case["actions_after_viability"] = jnp.zeros(3, dtype=jnp.int32)
    reproduction, _, events = _resolve_and_mutate(case)
    assert int(reproduction.birth_count) == 0
    assert not bool(jnp.any(events.born))


def test_zero_mutation_rate_keeps_valid_zero_summary_event(reproduction_case):
    reproduction, mutation, events = _resolve_and_mutate(reproduction_case, rate=0.0)
    assert int(reproduction.birth_count) == 1
    assert int(jnp.sum(events.born)) == 1
    assert jnp.all(mutation.selected_parameter_count == 0)
    assert jnp.all(events.mutation_effective_count == 0)
    assert jnp.all(events.mutation_sum_abs_delta == 0)
    assert jnp.all(events.mutation_max_abs_delta == 0)


def test_event_builder_is_eager_jit_and_scan_equivalent_without_rng(reproduction_case):
    reproduction, mutation, eager = _resolve_and_mutate(reproduction_case)
    arguments = {
        "population_before_reproduction": reproduction_case["population"],
        "reproduction": reproduction,
        "mutation": mutation,
        "step": reproduction_case["step"],
    }
    compiled = eqx.filter_jit(build_birth_events)(**arguments)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, eager, compiled))

    def body(carry, current_step):
        event = build_birth_events(**{**arguments, "step": current_step})
        return carry + jnp.sum(event.born, dtype=COUNT_DTYPE), event.child_agent_id

    count, child_ids = jax.lax.scan(
        body, jnp.asarray(0, dtype=COUNT_DTYPE), jnp.asarray([12], dtype=COUNT_DTYPE)
    )
    assert int(count) == 1
    assert jnp.array_equal(child_ids[0], eager.child_agent_id)


def test_builder_does_not_modify_inputs(reproduction_case):
    reproduction, mutation, _ = _resolve_and_mutate(reproduction_case)
    population_before = jax.tree.map(jnp.copy, reproduction_case["population"])
    population_after = jax.tree.map(jnp.copy, reproduction.population)
    genomes_after = jax.tree.map(jnp.copy, mutation.genomes)
    build_birth_events(
        population_before_reproduction=reproduction_case["population"],
        reproduction=reproduction,
        mutation=mutation,
        step=reproduction_case["step"],
    )
    assert jax.tree.all(
        jax.tree.map(jnp.array_equal, population_before, reproduction_case["population"])
    )
    assert jax.tree.all(jax.tree.map(jnp.array_equal, population_after, reproduction.population))
    assert jax.tree.all(jax.tree.map(jnp.array_equal, genomes_after, mutation.genomes))
    assert not hasattr(PopulationState, "offspring_count")
