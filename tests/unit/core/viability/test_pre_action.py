# ruff: noqa: ANN001, ANN003, ANN201, ANN202
import inspect

import jax
import jax.numpy as jnp
import pytest

from evolucio.core.codes import DeathCauseCode
from evolucio.core.dtypes import COUNT_DTYPE, REAL_DTYPE, STEP_DTYPE
from evolucio.core.energy import apply_basal_metabolism_and_age
from evolucio.core.state import PopulationState
from evolucio.core.viability import resolve_pre_action_viability


def resolve(population, genomes, world):
    return resolve_pre_action_viability(
        population=population,
        genomes=genomes,
        world=world,
        step=jnp.asarray(8, dtype=STEP_DTYPE),
        death_energy_threshold=jnp.asarray(0, dtype=REAL_DTYPE),
        maximum_age=jnp.asarray(10, dtype=COUNT_DTYPE),
        width=2,
        height=2,
    )


def changed(population, **changes):
    values = vars(population).copy()
    for field, value in changes.items():
        array = values[field]
        values[field] = array.at[0].set(jnp.asarray(value, dtype=array.dtype))
    return PopulationState(**values)


@pytest.mark.parametrize(
    ("energy", "dies", "cause"),
    [
        (0.1, False, DeathCauseCode.NONE),
        (0.0, True, DeathCauseCode.ENERGY_DEPLETION),
        (-0.1, True, DeathCauseCode.ENERGY_DEPLETION),
        (float("nan"), True, DeathCauseCode.INVALID_STATE),
        (float("inf"), True, DeathCauseCode.INVALID_STATE),
    ],
)
def test_energy_boundary_and_nonfinite_priority(viability_state, energy, dies, cause):
    population, genomes, world = viability_state
    result = resolve(changed(population, energy=energy), genomes, world)
    assert bool(result.deaths.died[0]) is dies
    assert int(result.deaths.cause[0]) == int(cause)


@pytest.mark.parametrize(
    ("changes", "cause"),
    [
        ({"age": 9}, DeathCauseCode.NONE),
        ({"age": 10}, DeathCauseCode.MAX_AGE),
        ({"age": 10, "energy": 0.0}, DeathCauseCode.ENERGY_DEPLETION),
        ({"age": -1, "energy": 0.0}, DeathCauseCode.INVALID_STATE),
        ({"position": [2, 0]}, DeathCauseCode.INVALID_STATE),
        ({"agent_id": -1}, DeathCauseCode.INVALID_STATE),
        ({"lineage_id": -1}, DeathCauseCode.INVALID_STATE),
        ({"genome_id": -1}, DeathCauseCode.INVALID_STATE),
        ({"generation": -1}, DeathCauseCode.INVALID_STATE),
        ({"birth_step": 9}, DeathCauseCode.INVALID_STATE),
    ],
)
def test_age_invalid_state_and_cause_priority(viability_state, changes, cause):
    population, genomes, world = viability_state
    result = resolve(changed(population, **changes), genomes, world)
    assert int(result.deaths.cause[0]) == int(cause)


def test_cleanup_genomes_occupancy_and_survivor_preservation(viability_state):
    population, genomes, world = viability_state
    result = resolve(changed(population, energy=0.0), genomes, world)
    assert result.population.alive.tolist() == [False, True, False]
    assert result.population.agent_id.tolist() == [-1, 11, -1]
    assert result.population.position.tolist() == [[-1, -1], [1, 1], [-1, -1]]
    assert result.population.energy.tolist() == [0.0, 5.0, 0.0]
    assert result.population.age.tolist() == [0, 3, 0]
    assert all(bool(jnp.all(leaf[0] == 0)) for leaf in jax.tree.leaves(result.genomes))
    assert all(
        bool(jnp.array_equal(a[1], b[1]))
        for a, b in zip(jax.tree.leaves(result.genomes), jax.tree.leaves(genomes), strict=True)
    )
    assert result.world.occupancy.tolist() == [[0, 0], [0, 1]]
    assert jnp.array_equal(result.world.resources, world.resources)
    assert jnp.array_equal(result.world.environment, world.environment)
    assert int(result.invalid_alive_position_count_after) == 0
    assert int(result.death_count) == int(result.energy_death_count) == 1


def test_inactive_slot_environment_and_shared_cell_do_not_kill(viability_state):
    population, genomes, world = viability_state
    values = vars(population).copy()
    values["position"] = population.position.at[1].set(
        jnp.asarray([0, 0], dtype=population.position.dtype)
    )
    result = resolve(PopulationState(**values), genomes, world)
    assert result.deaths.died.tolist() == [False, False, False]
    assert int(result.world.occupancy[0, 0]) == 2


def test_pr19_integration_energy_and_age(viability_state):
    population, genomes, world = viability_state
    values = vars(population).copy()
    values["energy"] = population.energy.at[0].set(1.0)
    values["age"] = population.age.at[1].set(9)
    metabolised = apply_basal_metabolism_and_age(
        PopulationState(**values), basal_metabolic_cost=jnp.asarray(1.0, dtype=REAL_DTYPE)
    ).population
    result = resolve(metabolised, genomes, world)
    assert result.deaths.cause.tolist()[:2] == [
        int(DeathCauseCode.ENERGY_DEPLETION),
        int(DeathCauseCode.MAX_AGE),
    ]


def test_eager_jit_scan_fixed_shapes_and_no_rng(viability_state):
    population, genomes, world = viability_state
    eager = resolve(population, genomes, world)
    compiled = jax.jit(resolve)(population, genomes, world)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, eager, compiled))
    assert eager.deaths.died.shape == (3,)
    assert eager.deaths.position.shape == (3, 2)

    def body(carry, _):
        pop, genes, current_world = carry
        result = resolve(pop, genes, current_world)
        return (result.population, result.genomes, result.world), result.death_count

    _, counts = jax.lax.scan(body, (population, genomes, world), xs=None, length=2)
    assert counts.shape == (2,)
    assert "key" not in inspect.signature(resolve_pre_action_viability).parameters
