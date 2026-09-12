# ruff: noqa: ANN201

import jax
import jax.numpy as jnp
import pytest

from evolucio.core.codes import ActionCode
from evolucio.core.dtypes import CODE_DTYPE, COUNT_DTYPE, ID_DTYPE, INDEX_DTYPE, REAL_DTYPE
from evolucio.core.ids import create_id_counters
from evolucio.core.policy import create_empty_genome_batch
from evolucio.core.spatial import rebuild_world_occupancy
from evolucio.core.state import PopulationState, WorldState
from evolucio.core.viability import ReproductionGateResult


@pytest.fixture
def reproduction_case():
    population = PopulationState(
        alive=jnp.asarray([True, False, False]),
        agent_id=jnp.asarray([7, -1, -1], dtype=ID_DTYPE),
        parent_id=jnp.asarray([-1, -1, -1], dtype=ID_DTYPE),
        lineage_id=jnp.asarray([4, -1, -1], dtype=ID_DTYPE),
        genome_id=jnp.asarray([9, -1, -1], dtype=ID_DTYPE),
        generation=jnp.asarray([2, 0, 0], dtype=COUNT_DTYPE),
        position=jnp.asarray([[1, 1], [-1, -1], [-1, -1]], dtype=INDEX_DTYPE),
        energy=jnp.asarray([10, 0, 0], dtype=REAL_DTYPE),
        birth_step=jnp.asarray([0, 0, 0], dtype=COUNT_DTYPE),
        age=jnp.asarray([5, 0, 0], dtype=COUNT_DTYPE),
    )
    genomes = jax.tree.map(
        lambda leaf: leaf.at[0].set(jnp.arange(leaf[0].size).reshape(leaf[0].shape)),
        create_empty_genome_batch(3),
    )
    world = WorldState(
        resources=jnp.arange(9, dtype=REAL_DTYPE).reshape(3, 3),
        environment=jnp.full((3, 3), 0.5, dtype=REAL_DTYPE),
        occupancy=jnp.zeros((3, 3), dtype=COUNT_DTYPE),
    )
    world = rebuild_world_occupancy(world, population, width=3, height=3).world
    gate = ReproductionGateResult(
        eligible=jnp.asarray([True, False, False]),
        gate_codes=jnp.zeros(3, dtype=CODE_DTYPE),
        projected_parent_energy=jnp.asarray([6, 0, 0], dtype=REAL_DTYPE),
        requested_count=jnp.asarray(1, dtype=COUNT_DTYPE),
        eligible_count=jnp.asarray(1, dtype=COUNT_DTYPE),
        suicidal_block_count=jnp.asarray(0, dtype=COUNT_DTYPE),
    )
    return {
        "population": population,
        "genomes": genomes,
        "world": world,
        "ids": create_id_counters(next_agent_id=20, next_genome_id=30, next_lineage_id=8),
        "reproduction_gate": gate,
        "actions_after_viability": jnp.asarray(
            [ActionCode.REPRODUCE, ActionCode.STAY, ActionCode.STAY], dtype=CODE_DTYPE
        ),
        "step": jnp.asarray(12, dtype=COUNT_DTYPE),
        "reproduction_energy_cost": jnp.asarray(4, dtype=REAL_DTYPE),
        "offspring_initial_energy": jnp.asarray(3, dtype=REAL_DTYPE),
        "birth_placement_key": jax.random.key(1),
        "reproduction_conflict_key": jax.random.key(2),
        "death_energy_threshold": jnp.asarray(0, dtype=REAL_DTYPE),
        "width": 3,
        "height": 3,
    }
