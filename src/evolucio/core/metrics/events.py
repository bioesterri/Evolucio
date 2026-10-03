"""Fixed-capacity event buffers for one completed simulation step."""

# pyright: reportUnknownMemberType=false

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.dtypes import CODE_DTYPE, COUNT_DTYPE, MASK_DTYPE
from evolucio.core.evolution import BirthEventBatch
from evolucio.core.types import Array
from evolucio.core.viability import DeathRecordBatch


class DeathPhaseCode(IntEnum):
    """Stable phase in which a death was observed."""

    NONE = 0
    PRE_ACTION = 1
    POST_ACTION = 2


class DeathEventBuffer(eqx.Module):
    """Compacted death records with population capacity and explicit overflow."""

    records: DeathRecordBatch
    phase: Array
    count: Array
    overflow: Array


class StepEventBuffer(eqx.Module):
    """Individual birth and death events emitted by one completed step."""

    births: BirthEventBatch
    deaths: DeathEventBuffer


def _compact_field(values: Array, valid: Array, capacity: int, fill: int | float) -> Array:
    ranks = jnp.cumsum(valid.astype(COUNT_DTYPE), dtype=COUNT_DTYPE) - 1
    safe_ranks = jnp.where(valid, ranks, capacity)
    output_shape = (capacity, *values.shape[1:])
    output = jnp.full(output_shape, fill, dtype=values.dtype)
    return output.at[safe_ranks].set(values, mode="drop")


def combine_death_records(
    *, pre_action: DeathRecordBatch, post_action: DeathRecordBatch
) -> DeathEventBuffer:
    """Deduplicate and compact both mortality phases into fixed capacity."""
    capacity = pre_action.died.shape[0]
    pre_valid = pre_action.died
    duplicated_post = jnp.any(
        post_action.died[:, None]
        & pre_valid[None, :]
        & (post_action.agent_id[:, None] == pre_action.agent_id[None, :]),
        axis=1,
    )
    post_valid = post_action.died & ~duplicated_post
    valid = jnp.concatenate((pre_valid, post_valid))
    raw_count = jnp.sum(pre_action.died, dtype=COUNT_DTYPE) + jnp.sum(
        post_action.died, dtype=COUNT_DTYPE
    )
    count = jnp.minimum(jnp.sum(valid, dtype=COUNT_DTYPE), capacity).astype(COUNT_DTYPE)
    phases = jnp.concatenate(
        (
            jnp.full((capacity,), DeathPhaseCode.PRE_ACTION, dtype=CODE_DTYPE),
            jnp.full((capacity,), DeathPhaseCode.POST_ACTION, dtype=CODE_DTYPE),
        )
    )

    def merged(name: str, fill: int | float) -> Array:
        values = jnp.concatenate((getattr(pre_action, name), getattr(post_action, name)), axis=0)
        return _compact_field(values, valid, capacity, fill)

    records = DeathRecordBatch(
        died=_compact_field(valid.astype(MASK_DTYPE), valid, capacity, False),
        agent_id=merged("agent_id", -1),
        parent_id=merged("parent_id", -1),
        lineage_id=merged("lineage_id", -1),
        genome_id=merged("genome_id", -1),
        generation=merged("generation", 0),
        death_step=merged("death_step", 0),
        age=merged("age", 0),
        energy=merged("energy", 0.0),
        position=merged("position", -1),
        cause=merged("cause", 0),
    )
    return DeathEventBuffer(
        records=records,
        phase=_compact_field(phases, valid, capacity, DeathPhaseCode.NONE).astype(CODE_DTYPE),
        count=count,
        overflow=(raw_count > capacity).astype(MASK_DTYPE),
    )


def create_step_event_buffer(
    *,
    births: BirthEventBatch,
    pre_action_deaths: DeathRecordBatch,
    post_action_deaths: DeathRecordBatch,
) -> StepEventBuffer:
    """Build the fixed-shape event output for a completed step."""
    return StepEventBuffer(
        births=births,
        deaths=combine_death_records(pre_action=pre_action_deaths, post_action=post_action_deaths),
    )
