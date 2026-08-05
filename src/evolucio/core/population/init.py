"""Pure construction of fixed-capacity founder population arrays."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.codes import RngStreamCode
from evolucio.core.dtypes import (
    COUNT_DTYPE,
    ID_DTYPE,
    INDEX_DTYPE,
    MASK_DTYPE,
    REAL_DTYPE,
    STEP_DTYPE,
)
from evolucio.core.ids import NULL_ID, IdCounters, allocate_ids
from evolucio.core.rng import derive_indexed_key, derive_stream_key
from evolucio.core.state import PopulationState, WorldState
from evolucio.core.types import Array
from evolucio.core.world import rebuild_world_occupancy

if TYPE_CHECKING:
    from evolucio.config.compile import EnergyCoreConfig, PopulationCoreConfig

INACTIVE_POSITION_COORDINATE = -1
INITIAL_GENERATION = 0
INITIAL_AGE = 0
INITIAL_BIRTH_STEP = 0
_INITIAL_POSITION_X_SUBSTREAM = 0
_INITIAL_POSITION_Y_SUBSTREAM = 1


class PopulationInitializationResult(eqx.Module):
    """Atomic result of founder initialization and initial occupancy."""

    population: PopulationState
    world: WorldState
    ids: IdCounters
    overflow: Array


class _FounderIdAllocation(eqx.Module):
    agent_ids: Array
    genome_ids: Array
    lineage_ids: Array
    ids: IdCounters
    overflow: Array


def create_empty_population(max_agents: int) -> PopulationState:
    """Create the canonical inactive representation for every population slot."""
    if isinstance(max_agents, bool) or not isinstance(max_agents, int):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("max_agents must be a Python int, not bool")
    if max_agents <= 0:
        raise ValueError("max_agents must be positive")
    shape = (max_agents,)
    return PopulationState(
        alive=jnp.zeros(shape, dtype=MASK_DTYPE),
        agent_id=jnp.full(shape, NULL_ID, dtype=ID_DTYPE),
        parent_id=jnp.full(shape, NULL_ID, dtype=ID_DTYPE),
        lineage_id=jnp.full(shape, NULL_ID, dtype=ID_DTYPE),
        genome_id=jnp.full(shape, NULL_ID, dtype=ID_DTYPE),
        generation=jnp.full(shape, INITIAL_GENERATION, dtype=COUNT_DTYPE),
        position=jnp.full((max_agents, 2), INACTIVE_POSITION_COORDINATE, dtype=INDEX_DTYPE),
        energy=jnp.zeros(shape, dtype=REAL_DTYPE),
        birth_step=jnp.full(shape, INITIAL_BIRTH_STEP, dtype=STEP_DTYPE),
        age=jnp.full(shape, INITIAL_AGE, dtype=COUNT_DTYPE),
    )


def _build_initial_alive_mask(*, initial_agents: Array, max_agents: int) -> Array:
    slot_indices = jnp.arange(max_agents, dtype=INDEX_DTYPE)
    return (slot_indices < initial_agents).astype(MASK_DTYPE)


def _allocate_founder_ids(ids: IdCounters, alive: Array) -> _FounderIdAllocation:
    agents = allocate_ids(ids.next_agent_id, alive)
    genomes = allocate_ids(ids.next_genome_id, alive)
    lineages = allocate_ids(ids.next_lineage_id, alive)
    overflow = (agents.overflow | genomes.overflow | lineages.overflow).astype(MASK_DTYPE)
    nulls = jnp.full_like(agents.values, NULL_ID)
    return _FounderIdAllocation(
        agent_ids=jnp.where(overflow, nulls, agents.values),
        genome_ids=jnp.where(overflow, nulls, genomes.values),
        lineage_ids=jnp.where(overflow, nulls, lineages.values),
        ids=IdCounters(
            next_agent_id=jnp.where(overflow, ids.next_agent_id, agents.next_id),
            next_genome_id=jnp.where(overflow, ids.next_genome_id, genomes.next_id),
            next_lineage_id=jnp.where(overflow, ids.next_lineage_id, lineages.next_id),
        ),
        overflow=overflow,
    )


def _sample_initial_positions(
    *, key: Array, max_agents: int, width: int, height: int, alive: Array
) -> Array:
    x_key = derive_indexed_key(key, _INITIAL_POSITION_X_SUBSTREAM)
    y_key = derive_indexed_key(key, _INITIAL_POSITION_Y_SUBSTREAM)
    x = jax.random.randint(x_key, (max_agents,), 0, width, dtype=INDEX_DTYPE)
    y = jax.random.randint(y_key, (max_agents,), 0, height, dtype=INDEX_DTYPE)
    sampled = jnp.stack((x, y), axis=1)
    inactive = jnp.asarray(INACTIVE_POSITION_COORDINATE, dtype=INDEX_DTYPE)
    return jnp.where(alive[:, None], sampled, inactive)


def initialize_population(
    world: WorldState,
    population_config: PopulationCoreConfig,
    energy_config: EnergyCoreConfig,
    root_key: Array,
    ids: IdCounters,
) -> PopulationInitializationResult:
    """Initialize founders atomically without advancing the persistent root key."""
    if population_config.placement != "random":
        raise ValueError(f"unsupported initial placement: {population_config.placement}")
    max_agents = population_config.max_agents
    height, width = world.resources.shape
    alive = _build_initial_alive_mask(
        initial_agents=population_config.initial_agents, max_agents=max_agents
    )
    allocation = _allocate_founder_ids(ids, alive)
    population_key = derive_stream_key(root_key, RngStreamCode.AGENT_INITIALIZATION)
    positions = _sample_initial_positions(
        key=population_key,
        max_agents=max_agents,
        width=width,
        height=height,
        alive=alive,
    )
    successful_population = PopulationState(
        alive=alive,
        agent_id=allocation.agent_ids,
        parent_id=jnp.full((max_agents,), NULL_ID, dtype=ID_DTYPE),
        lineage_id=allocation.lineage_ids,
        genome_id=allocation.genome_ids,
        generation=jnp.full((max_agents,), INITIAL_GENERATION, dtype=COUNT_DTYPE),
        position=positions,
        energy=jnp.where(alive, energy_config.initial_energy, 0).astype(REAL_DTYPE),
        birth_step=jnp.full((max_agents,), INITIAL_BIRTH_STEP, dtype=STEP_DTYPE),
        age=jnp.full((max_agents,), INITIAL_AGE, dtype=COUNT_DTYPE),
    )
    successful_world = rebuild_world_occupancy(
        world, successful_population, width=width, height=height
    ).world
    empty = create_empty_population(max_agents)

    def select_success(success: Array, failure: Array) -> Array:
        return jnp.where(allocation.overflow, failure, success)

    population = jax.tree.map(select_success, successful_population, empty)
    updated_world = jax.tree.map(select_success, successful_world, world)
    return PopulationInitializationResult(
        population=population,
        world=updated_world,
        ids=allocation.ids,
        overflow=allocation.overflow,
    )
