"""Pure, vectorised pre-action viability resolution."""

# pyright: reportUnknownMemberType=false

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.codes import DeathCauseCode
from evolucio.core.dtypes import CODE_DTYPE, COUNT_DTYPE, ID_DTYPE, INDEX_DTYPE, REAL_DTYPE
from evolucio.core.ids import NULL_ID
from evolucio.core.policy.batch import GenomeBatch
from evolucio.core.population.init import INACTIVE_POSITION_COORDINATE
from evolucio.core.spatial import rebuild_world_occupancy
from evolucio.core.state import PopulationState, WorldState
from evolucio.core.types import Array
from evolucio.core.world.bounds import positions_in_bounds

from .death import DeathRecordBatch, build_death_records


class PreActionViabilityResult(eqx.Module):
    """Updated state, death records, and fixed-shape viability diagnostics."""

    population: PopulationState
    genomes: GenomeBatch
    world: WorldState
    deaths: DeathRecordBatch
    death_count: Array
    invalid_state_death_count: Array
    energy_death_count: Array
    max_age_death_count: Array
    invalid_alive_position_count_after: Array


def _canonicalize_population(population: PopulationState, dies: Array) -> PopulationState:
    alive = population.alive & ~dies
    null_id = jnp.asarray(NULL_ID, dtype=ID_DTYPE)
    return PopulationState(
        alive=alive,
        agent_id=jnp.where(dies, null_id, population.agent_id),
        parent_id=jnp.where(dies, null_id, population.parent_id),
        lineage_id=jnp.where(dies, null_id, population.lineage_id),
        genome_id=jnp.where(dies, null_id, population.genome_id),
        generation=jnp.where(dies, 0, population.generation),
        position=jnp.where(
            dies[:, None],
            jnp.asarray(INACTIVE_POSITION_COORDINATE, dtype=INDEX_DTYPE),
            population.position,
        ),
        energy=jnp.where(dies, jnp.asarray(0, dtype=REAL_DTYPE), population.energy),
        birth_step=jnp.where(dies, 0, population.birth_step),
        age=jnp.where(dies, 0, population.age),
    )


def _canonicalize_genomes(genomes: GenomeBatch, dies: Array) -> GenomeBatch:
    def clear_dead_slots(leaf: Array) -> Array:
        broadcast = dies.reshape((dies.shape[0],) + (1,) * (leaf.ndim - 1))
        return jnp.where(broadcast, jnp.asarray(0, dtype=leaf.dtype), leaf)

    return jax.tree.map(clear_dead_slots, genomes)


def resolve_pre_action_viability(
    *,
    population: PopulationState,
    genomes: GenomeBatch,
    world: WorldState,
    step: Array,
    death_energy_threshold: Array,
    maximum_age: Array,
    width: int,
    height: int,
) -> PreActionViabilityResult:
    """Remove non-viable live agents before observations and action inference."""
    alive = population.alive
    valid_position = positions_in_bounds(population.position, width=width, height=height)
    invalid_state = alive & (
        ~jnp.isfinite(population.energy)
        | (population.age < 0)
        | ~valid_position
        | (population.agent_id < 0)
        | (population.lineage_id < 0)
        | (population.genome_id < 0)
        | (population.generation < 0)
        | (population.birth_step < 0)
        | (population.birth_step > step)
    )
    energy_depleted = alive & (population.energy <= death_energy_threshold)
    max_age_reached = alive & (population.age >= maximum_age)

    # Exclusive masks encode priority explicitly, independently of cause assignment order.
    invalid_deaths = invalid_state
    energy_deaths = energy_depleted & ~invalid_state
    max_age_deaths = max_age_reached & ~invalid_state & ~energy_depleted
    dies = invalid_deaths | energy_deaths | max_age_deaths
    cause = (
        invalid_deaths.astype(CODE_DTYPE) * int(DeathCauseCode.INVALID_STATE)
        + energy_deaths.astype(CODE_DTYPE) * int(DeathCauseCode.ENERGY_DEPLETION)
        + max_age_deaths.astype(CODE_DTYPE) * int(DeathCauseCode.MAX_AGE)
    ).astype(CODE_DTYPE)

    deaths = build_death_records(population=population, dies=dies, cause=cause, step=step)
    population_after = _canonicalize_population(population, dies)
    genomes_after = _canonicalize_genomes(genomes, dies)
    occupancy_update = rebuild_world_occupancy(world, population_after, width=width, height=height)
    return PreActionViabilityResult(
        population=population_after,
        genomes=genomes_after,
        world=occupancy_update.world,
        deaths=deaths,
        death_count=jnp.sum(dies, dtype=COUNT_DTYPE),
        invalid_state_death_count=jnp.sum(invalid_deaths, dtype=COUNT_DTYPE),
        energy_death_count=jnp.sum(energy_deaths, dtype=COUNT_DTYPE),
        max_age_death_count=jnp.sum(max_age_deaths, dtype=COUNT_DTYPE),
        invalid_alive_position_count_after=occupancy_update.invalid_alive_count,
    )
