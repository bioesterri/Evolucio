"""Fixed-capacity transient death records."""

# pyright: reportUnknownMemberType=false

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.codes import DeathCauseCode
from evolucio.core.dtypes import CODE_DTYPE, ID_DTYPE, INDEX_DTYPE, REAL_DTYPE, STEP_DTYPE
from evolucio.core.ids import NULL_ID
from evolucio.core.policy.batch import GenomeBatch
from evolucio.core.state import INACTIVE_POSITION_COORDINATE, PopulationState
from evolucio.core.types import Array
from evolucio.core.world.bounds import positions_in_bounds


class DeathRecordBatch(eqx.Module):
    """One fixed-shape death-record slot per population slot."""

    died: Array
    agent_id: Array
    parent_id: Array
    lineage_id: Array
    genome_id: Array
    generation: Array
    death_step: Array
    age: Array
    energy: Array
    position: Array
    cause: Array


class DeathEvaluation(eqx.Module):
    """Exclusive fixed-shape death masks and their prioritized cause."""

    dies: Array
    cause: Array
    invalid_deaths: Array
    energy_deaths: Array
    max_age_deaths: Array


def evaluate_deaths(
    *,
    population: PopulationState,
    step: Array,
    death_energy_threshold: Array,
    maximum_age: Array,
    width: int,
    height: int,
) -> DeathEvaluation:
    """Apply the shared INVALID_STATE, energy, then age mortality contract."""
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
    invalid_deaths = invalid_state
    energy_deaths = energy_depleted & ~invalid_state
    max_age_deaths = max_age_reached & ~invalid_state & ~energy_depleted
    dies = invalid_deaths | energy_deaths | max_age_deaths
    cause = (
        invalid_deaths.astype(CODE_DTYPE) * int(DeathCauseCode.INVALID_STATE)
        + energy_deaths.astype(CODE_DTYPE) * int(DeathCauseCode.ENERGY_DEPLETION)
        + max_age_deaths.astype(CODE_DTYPE) * int(DeathCauseCode.MAX_AGE)
    ).astype(CODE_DTYPE)
    return DeathEvaluation(
        dies=dies,
        cause=cause,
        invalid_deaths=invalid_deaths,
        energy_deaths=energy_deaths,
        max_age_deaths=max_age_deaths,
    )


def build_death_records(
    *, population: PopulationState, dies: Array, cause: Array, step: Array
) -> DeathRecordBatch:
    """Snapshot dying agents while canonicalising all non-death records."""
    null_id = jnp.asarray(NULL_ID, dtype=ID_DTYPE)
    zero_step = jnp.asarray(0, dtype=STEP_DTYPE)
    return DeathRecordBatch(
        died=dies,
        agent_id=jnp.where(dies, population.agent_id, null_id),
        parent_id=jnp.where(dies, population.parent_id, null_id),
        lineage_id=jnp.where(dies, population.lineage_id, null_id),
        genome_id=jnp.where(dies, population.genome_id, null_id),
        generation=jnp.where(dies, population.generation, 0),
        death_step=jnp.where(dies, step, zero_step).astype(STEP_DTYPE),
        age=jnp.where(dies, population.age, 0),
        energy=jnp.where(dies, population.energy, 0).astype(REAL_DTYPE),
        position=jnp.where(
            dies[:, None],
            population.position,
            jnp.asarray(INACTIVE_POSITION_COORDINATE, dtype=INDEX_DTYPE),
        ),
        cause=jnp.where(dies, cause, jnp.asarray(DeathCauseCode.NONE, dtype=CODE_DTYPE)).astype(
            CODE_DTYPE
        ),
    )


def canonicalize_population(population: PopulationState, dies: Array) -> PopulationState:
    """Replace dying slots with the one canonical inactive representation."""
    null_id = jnp.asarray(NULL_ID, dtype=ID_DTYPE)
    return PopulationState(
        alive=population.alive & ~dies,
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


def canonicalize_genomes(genomes: GenomeBatch, dies: Array) -> GenomeBatch:
    """Zero every genome leaf belonging to a dying slot."""

    def clear_dead_slots(leaf: Array) -> Array:
        broadcast = dies.reshape((dies.shape[0],) + (1,) * (leaf.ndim - 1))
        return jnp.where(broadcast, jnp.asarray(0, dtype=leaf.dtype), leaf)

    return jax.tree.map(clear_dead_slots, genomes)
