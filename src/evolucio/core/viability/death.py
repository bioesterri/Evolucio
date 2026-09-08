"""Fixed-capacity transient death records."""

# pyright: reportUnknownMemberType=false

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.codes import DeathCauseCode
from evolucio.core.dtypes import CODE_DTYPE, ID_DTYPE, INDEX_DTYPE, REAL_DTYPE, STEP_DTYPE
from evolucio.core.ids import NULL_ID
from evolucio.core.population.init import INACTIVE_POSITION_COORDINATE
from evolucio.core.state import PopulationState
from evolucio.core.types import Array


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
