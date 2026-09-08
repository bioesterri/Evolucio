# ruff: noqa: ANN001, ANN003, ANN201, ANN202
import inspect

import jax
import jax.numpy as jnp
import pytest

from evolucio.core.codes import ActionCode, DeathCauseCode
from evolucio.core.dtypes import CODE_DTYPE, COUNT_DTYPE, REAL_DTYPE, STEP_DTYPE
from evolucio.core.state import PopulationState
from evolucio.core.viability import resolve_post_action_viability


def changed(population, slot=0, **changes):
    values = vars(population).copy()
    for field, value in changes.items():
        values[field] = values[field].at[slot].set(jnp.asarray(value, dtype=values[field].dtype))
    return PopulationState(**values)


def resolve(population, genomes, world, actions=None):
    if actions is None:
        actions = jnp.asarray(
            [ActionCode.REPRODUCE, ActionCode.EAT, ActionCode.STAY], dtype=CODE_DTYPE
        )
    return resolve_post_action_viability(
        population=population,
        genomes=genomes,
        world=world,
        actions_after_feeding=actions,
        step=jnp.asarray(8, dtype=STEP_DTYPE),
        death_energy_threshold=jnp.asarray(0, dtype=REAL_DTYPE),
        maximum_age=jnp.asarray(10, dtype=COUNT_DTYPE),
        width=2,
        height=2,
    )


@pytest.mark.parametrize(
    ("energy", "dies", "cause"),
    [
        (0.1, False, DeathCauseCode.NONE),
        (0.0, True, DeathCauseCode.ENERGY_DEPLETION),
        (-0.1, True, DeathCauseCode.ENERGY_DEPLETION),
        (float("nan"), True, DeathCauseCode.INVALID_STATE),
    ],
)
def test_shared_mortality_boundaries(viability_state, energy, dies, cause):
    population, genomes, world = viability_state
    result = resolve(changed(population, energy=energy), genomes, world)
    assert bool(result.deaths.died[0]) is dies
    assert int(result.deaths.cause[0]) == int(cause)


def test_record_cleanup_actions_and_occupancy(viability_state):
    population, genomes, world = viability_state
    dying = changed(population, energy=0.0)
    result = resolve(dying, genomes, world)
    assert float(result.deaths.energy[0]) == 0.0
    assert int(result.deaths.agent_id[0]) == 10
    assert result.deaths.position[0].tolist() == [0, 0]
    assert result.population.alive.tolist() == [False, True, False]
    assert result.population.agent_id.tolist() == [-1, 11, -1]
    assert result.population.position.tolist()[0] == [-1, -1]
    assert all(bool(jnp.all(leaf[0] == 0)) for leaf in jax.tree.leaves(result.genomes))
    assert result.world.occupancy.tolist() == [[0, 0], [0, 1]]
    assert result.actions_after_viability.tolist() == [ActionCode.STAY, ActionCode.EAT, 0]
    assert result.actions_after_viability.dtype == jnp.dtype(CODE_DTYPE)
    assert int(result.death_count) == int(result.energy_death_count) == 1


def test_survivors_resources_and_environment_are_unchanged(viability_state):
    population, genomes, world = viability_state
    result = resolve(population, genomes, world)
    assert jnp.array_equal(result.population.energy, population.energy)
    assert jnp.array_equal(result.world.resources, world.resources)
    assert jnp.array_equal(result.world.environment, world.environment)


def test_eager_jit_scan_fixed_shapes_and_no_rng(viability_state):
    population, genomes, world = viability_state
    eager = resolve(population, genomes, world)
    compiled = jax.jit(resolve)(population, genomes, world)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, eager, compiled))
    assert eager.actions_after_viability.shape == (3,)
    assert eager.deaths.position.shape == (3, 2)

    def body(carry, _):
        result = resolve(carry[0], carry[1], carry[2])
        return (result.population, result.genomes, result.world), result.death_count

    _, counts = jax.lax.scan(body, (population, genomes, world), xs=None, length=2)
    assert counts.shape == (2,)
    assert "rng" not in inspect.signature(resolve_post_action_viability).parameters
