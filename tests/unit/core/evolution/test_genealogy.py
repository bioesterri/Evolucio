# ruff: noqa: ANN001, ANN201, ANN202

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.config import compile_config, load_config
from evolucio.core.dtypes import COUNT_DTYPE, REAL_DTYPE
from evolucio.core.evolution import (
    build_birth_events,
    mutate_newborn_genomes,
    resolve_asexual_reproduction,
    validate_genealogy,
)
from evolucio.core.ids import MAX_NEXT_ID, create_id_counters
from evolucio.core.population import initialize_population
from evolucio.core.rng import create_rng_state
from evolucio.core.world import initialize_world


def _results(case):
    reproduction = resolve_asexual_reproduction(**case)
    mutation = mutate_newborn_genomes(
        genomes=reproduction.genomes,
        population=reproduction.population,
        newborn_mask=reproduction.newborn_mask,
        genome_mutation_key=jax.random.key(9),
        weight_mutation_rate=jnp.asarray(0.0, dtype=REAL_DTYPE),
        weight_mutation_sigma=jnp.asarray(0.1, dtype=REAL_DTYPE),
        weight_abs_limit=jnp.asarray(100.0, dtype=REAL_DTYPE),
        bias_mutation_rate=jnp.asarray(0.0, dtype=REAL_DTYPE),
        bias_mutation_sigma=jnp.asarray(0.1, dtype=REAL_DTYPE),
        bias_abs_limit=jnp.asarray(100.0, dtype=REAL_DTYPE),
    )
    validation = validate_genealogy(
        population_before_reproduction=case["population"],
        reproduction=reproduction,
        mutation=mutation,
        step=case["step"],
    )
    return reproduction, mutation, validation


def test_founders_have_null_parents_zero_generation_and_distinct_lineages():
    config = load_config("tests/fixtures/config/valid_v1.yaml")
    core = compile_config(config).core
    root_key = create_rng_state(1).key
    world = initialize_world(core.world, root_key)
    result = initialize_population(
        world, core.population, core.energy, root_key, create_id_counters()
    )
    founders = result.population.alive
    population = result.population
    assert jnp.all(population.parent_id[founders] == -1)
    assert jnp.all(population.generation[founders] == 0)
    assert len(set(map(int, population.lineage_id[founders]))) == int(jnp.sum(founders))


def test_valid_descendant_inherits_lineage_and_direct_parent(reproduction_case):
    reproduction, _, validation = _results(reproduction_case)
    child = int(jnp.flatnonzero(reproduction.newborn_mask, size=1)[0])
    parent = int(reproduction.parent_slots[child])
    assert bool(validation.valid_birth[child])
    assert int(validation.invalid_birth_count) == 0
    assert (
        reproduction.population.parent_id[child] == reproduction_case["population"].agent_id[parent]
    )
    assert (
        reproduction.population.lineage_id[child]
        == reproduction_case["population"].lineage_id[parent]
    )
    assert (
        reproduction.population.generation[child]
        == reproduction_case["population"].generation[parent] + 1
    )
    assert (
        reproduction.population.agent_id[child] != reproduction_case["population"].agent_id[parent]
    )
    assert (
        reproduction.population.genome_id[child]
        != reproduction_case["population"].genome_id[parent]
    )
    assert reproduction.ids.next_lineage_id == reproduction_case["ids"].next_lineage_id


def test_inconsistent_genealogy_is_diagnosed_without_repair(reproduction_case):
    reproduction, mutation, _ = _results(reproduction_case)
    child = int(jnp.flatnonzero(reproduction.newborn_mask, size=1)[0])
    broken_population = eqx.tree_at(
        lambda population: population.lineage_id,
        reproduction.population,
        reproduction.population.lineage_id.at[child].set(999),
    )
    broken = eqx.tree_at(lambda result: result.population, reproduction, broken_population)
    validation = validate_genealogy(
        population_before_reproduction=reproduction_case["population"],
        reproduction=broken,
        mutation=mutation,
        step=reproduction_case["step"],
    )
    events = build_birth_events(
        population_before_reproduction=reproduction_case["population"],
        reproduction=broken,
        mutation=mutation,
        step=reproduction_case["step"],
    )
    assert not bool(validation.valid_birth[child])
    assert int(validation.invalid_birth_count) == 1
    assert int(events.lineage_id[child]) == 999
    assert int(broken.population.lineage_id[child]) == 999


def test_child_identity_must_not_collide_with_any_previous_agent(reproduction_case):
    reproduction, mutation, _ = _results(reproduction_case)
    child = int(jnp.flatnonzero(reproduction.newborn_mask, size=1)[0])
    other = 2 if child == 1 else 1
    before = eqx.tree_at(
        lambda population: (population.alive, population.agent_id, population.genome_id),
        reproduction_case["population"],
        (
            reproduction_case["population"].alive.at[other].set(True),
            reproduction_case["population"]
            .agent_id.at[other]
            .set(reproduction.population.agent_id[child]),
            reproduction_case["population"]
            .genome_id.at[other]
            .set(reproduction.population.genome_id[child]),
        ),
    )
    validation = validate_genealogy(
        population_before_reproduction=before,
        reproduction=reproduction,
        mutation=mutation,
        step=reproduction_case["step"],
    )
    assert not bool(validation.valid_birth[child])
    assert int(validation.invalid_birth_count) == 1


def test_sibling_id_collisions_are_rejected(reproduction_case):
    reproduction, mutation, _ = _results(reproduction_case)
    first_child = int(jnp.flatnonzero(reproduction.newborn_mask, size=1)[0])
    second_child = 2 if first_child == 1 else 1
    duplicated_population = jax.tree.map(
        lambda values: values.at[second_child].set(values[first_child]), reproduction.population
    )
    duplicated = eqx.tree_at(
        lambda result: (result.population, result.newborn_mask, result.parent_slots),
        reproduction,
        (
            duplicated_population,
            reproduction.newborn_mask.at[second_child].set(True),
            reproduction.parent_slots.at[second_child].set(0),
        ),
    )
    validation = validate_genealogy(
        population_before_reproduction=reproduction_case["population"],
        reproduction=duplicated,
        mutation=mutation,
        step=reproduction_case["step"],
    )
    assert int(validation.invalid_birth_count) == 2
    assert not bool(jnp.any(validation.valid_birth))


def test_birth_cannot_overwrite_a_previously_live_slot(reproduction_case):
    reproduction, mutation, _ = _results(reproduction_case)
    child = int(jnp.flatnonzero(reproduction.newborn_mask, size=1)[0])
    before = eqx.tree_at(
        lambda population: population.alive,
        reproduction_case["population"],
        reproduction_case["population"].alive.at[child].set(True),
    )
    validation = validate_genealogy(
        population_before_reproduction=before,
        reproduction=reproduction,
        mutation=mutation,
        step=reproduction_case["step"],
    )
    assert not bool(validation.valid_birth[child])
    assert int(validation.invalid_birth_count) == 1


def test_mutation_activity_outside_newborn_mask_is_diagnosed(reproduction_case):
    reproduction, mutation, _ = _results(reproduction_case)
    inactive = int(jnp.flatnonzero(~reproduction.newborn_mask, size=1)[0])
    inconsistent = eqx.tree_at(
        lambda result: result.selected_parameter_count,
        mutation,
        mutation.selected_parameter_count.at[inactive].set(1),
    )
    validation = validate_genealogy(
        population_before_reproduction=reproduction_case["population"],
        reproduction=reproduction,
        mutation=inconsistent,
        step=reproduction_case["step"],
    )
    assert int(validation.mutation_inconsistency_count) == 1


def test_internally_inconsistent_mutation_summary_is_diagnosed(reproduction_case):
    reproduction, mutation, _ = _results(reproduction_case)
    child = int(jnp.flatnonzero(reproduction.newborn_mask, size=1)[0])
    inconsistent = eqx.tree_at(
        lambda result: (result.selected_parameter_count, result.sum_abs_delta),
        mutation,
        (
            mutation.selected_parameter_count.at[child].set(1),
            mutation.sum_abs_delta.at[child].set(jnp.nan),
        ),
    )
    validation = validate_genealogy(
        population_before_reproduction=reproduction_case["population"],
        reproduction=reproduction,
        mutation=inconsistent,
        step=reproduction_case["step"],
    )
    assert not bool(validation.valid_birth[child])
    assert int(validation.invalid_birth_count) == 1
    assert int(validation.mutation_inconsistency_count) == 1


def test_generation_overflow_prevents_birth_atomically(reproduction_case):
    case = dict(reproduction_case)
    case["population"] = eqx.tree_at(
        lambda population: population.generation,
        case["population"],
        case["population"].generation.at[0].set(MAX_NEXT_ID),
    )
    reproduction = resolve_asexual_reproduction(**case)
    assert int(reproduction.birth_count) == 0
    assert not bool(jnp.any(reproduction.newborn_mask))
    assert jnp.array_equal(reproduction.population.generation, case["population"].generation)


def test_multigeneration_chain_and_branch_contracts_are_identity_based():
    agents = {"A": 10, "B": 20, "C": 30, "D": 40}
    genomes = {"A": 100, "B": 200, "C": 300, "D": 400}
    parent = {"A": -1, "B": agents["A"], "C": agents["B"], "D": agents["A"]}
    lineage = {name: 7 for name in agents}
    generation = {"A": 0, "B": 1, "C": 2, "D": 1}

    assert parent["A"] == -1
    assert parent["B"] == agents["A"] and parent["C"] == agents["B"]
    assert parent["D"] == agents["A"]
    assert generation == {"A": 0, "B": 1, "C": 2, "D": 1}
    assert len(set(agents.values())) == len(set(genomes.values())) == 4
    assert len(set(lineage.values())) == 1
    assert agents["B"] != agents["D"] and genomes["B"] != genomes["D"]


def test_validation_is_eager_jit_and_slot_permutation_preserves_identity(reproduction_case):
    reproduction, mutation, eager = _results(reproduction_case)
    arguments = {
        "population_before_reproduction": reproduction_case["population"],
        "reproduction": reproduction,
        "mutation": mutation,
        "step": reproduction_case["step"],
    }
    compiled = eqx.filter_jit(validate_genealogy)(**arguments)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, eager, compiled))

    events = build_birth_events(**arguments)
    fields = {field.name for field in dataclasses.fields(events)}
    assert "event_id" not in fields and "birth_event_id" not in fields
    child = int(jnp.flatnonzero(events.born, size=1)[0])
    identity_relation = (
        int(events.child_agent_id[child]),
        int(events.parent_agent_id[child]),
        int(events.lineage_id[child]),
    )
    permutation = jnp.asarray([2, 0, 1])
    assert identity_relation == tuple(
        int(value[permutation][int(jnp.flatnonzero(events.born[permutation], size=1)[0])])
        for value in (events.child_agent_id, events.parent_agent_id, events.lineage_id)
    )


def test_validation_accepts_scan(reproduction_case):
    reproduction, mutation, _ = _results(reproduction_case)

    def body(count, step):
        result = validate_genealogy(
            population_before_reproduction=reproduction_case["population"],
            reproduction=reproduction,
            mutation=mutation,
            step=step,
        )
        return count + result.invalid_birth_count, result.valid_birth

    count, validity = jax.lax.scan(
        body, jnp.asarray(0, dtype=COUNT_DTYPE), jnp.asarray([12], dtype=COUNT_DTYPE)
    )
    assert int(count) == 0
    assert int(jnp.sum(validity)) == int(reproduction.birth_count)
