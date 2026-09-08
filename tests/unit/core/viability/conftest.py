# ruff: noqa: ANN201
import jax
import jax.numpy as jnp
import pytest

from evolucio.core.dtypes import COUNT_DTYPE, ID_DTYPE, INDEX_DTYPE, REAL_DTYPE, STEP_DTYPE
from evolucio.core.policy import create_empty_genome_batch
from evolucio.core.state import PopulationState, WorldState


@pytest.fixture
def viability_state():
    population = PopulationState(
        alive=jnp.asarray([True, True, False]),
        agent_id=jnp.asarray([10, 11, -1], dtype=ID_DTYPE),
        parent_id=jnp.asarray([-1, 10, -1], dtype=ID_DTYPE),
        lineage_id=jnp.asarray([20, 20, -1], dtype=ID_DTYPE),
        genome_id=jnp.asarray([30, 31, -1], dtype=ID_DTYPE),
        generation=jnp.asarray([0, 1, 0], dtype=COUNT_DTYPE),
        position=jnp.asarray([[0, 0], [1, 1], [-1, -1]], dtype=INDEX_DTYPE),
        energy=jnp.asarray([5.0, 5.0, 0.0], dtype=REAL_DTYPE),
        birth_step=jnp.asarray([0, 1, 0], dtype=STEP_DTYPE),
        age=jnp.asarray([2, 3, 0], dtype=COUNT_DTYPE),
    )
    genomes = jax.tree.map(lambda leaf: leaf.at[:2].set(2), create_empty_genome_batch(3))
    world = WorldState(
        resources=jnp.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=REAL_DTYPE),
        environment=jnp.full((2, 2), 0.9, dtype=REAL_DTYPE),
        occupancy=jnp.asarray([[1, 0], [0, 1]], dtype=COUNT_DTYPE),
    )
    return population, genomes, world
