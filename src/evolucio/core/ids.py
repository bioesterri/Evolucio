"""Persistent counters and vectorised internal identifier allocation."""

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp

from .dtypes import COUNT_DTYPE, ID_DTYPE, MASK_DTYPE
from .types import Array

NULL_ID = -1
FIRST_ID = 0
MAX_NEXT_ID = 2_147_483_647


class IdCounters(eqx.Module):
    """Next unassigned identifiers for the persistent identity domains."""

    next_agent_id: Array
    next_genome_id: Array
    next_lineage_id: Array


class IdAllocation(eqx.Module):
    """Atomic fixed-shape result of an identifier allocation request."""

    values: Array
    next_id: Array
    count: Array
    overflow: Array


def _counter(value: int, name: str) -> Array:
    if isinstance(value, bool) or not isinstance(value, int):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"{name} must be a Python int, not bool")
    if not FIRST_ID <= value <= MAX_NEXT_ID:
        raise ValueError(f"{name} must be between FIRST_ID and MAX_NEXT_ID")
    return jnp.asarray(value, dtype=ID_DTYPE)  # pyright: ignore[reportUnknownMemberType]


def create_id_counters(
    *,
    next_agent_id: int = FIRST_ID,
    next_genome_id: int = FIRST_ID,
    next_lineage_id: int = FIRST_ID,
) -> IdCounters:
    """Create scalar counters, optionally at a controlled restoration point."""
    return IdCounters(
        next_agent_id=_counter(next_agent_id, "next_agent_id"),
        next_genome_id=_counter(next_genome_id, "next_genome_id"),
        next_lineage_id=_counter(next_lineage_id, "next_lineage_id"),
    )


def allocate_ids(next_id: Array, request_mask: Array) -> IdAllocation:
    """Allocate monotonically increasing IDs atomically in mask order."""
    if next_id.shape != () or next_id.dtype != ID_DTYPE:
        raise TypeError("next_id must be a scalar with ID_DTYPE")
    if request_mask.ndim != 1 or request_mask.dtype != MASK_DTYPE:
        raise TypeError("request_mask must be one-dimensional with MASK_DTYPE")

    requested = request_mask.astype(COUNT_DTYPE)
    count = jnp.sum(requested, dtype=COUNT_DTYPE)
    remaining = (
        jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
            MAX_NEXT_ID, dtype=ID_DTYPE
        )
        - next_id
    )
    overflow = count > remaining
    safe_next_id = jnp.where(
        overflow,
        jnp.asarray(FIRST_ID, dtype=ID_DTYPE),  # pyright: ignore[reportUnknownMemberType]
        next_id,
    )
    offsets = jnp.cumsum(requested, dtype=ID_DTYPE) - jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
        1, dtype=ID_DTYPE
    )
    candidates = safe_next_id + offsets
    assigned = jnp.where(
        request_mask,
        candidates,
        jnp.asarray(NULL_ID, dtype=ID_DTYPE),  # pyright: ignore[reportUnknownMemberType]
    )
    values = jnp.where(
        overflow,
        jnp.full_like(assigned, NULL_ID),  # pyright: ignore[reportUnknownMemberType]
        assigned,
    )
    candidate_next_id = safe_next_id + count
    result_next_id = jnp.where(overflow, next_id, candidate_next_id)
    return IdAllocation(
        values=values,
        next_id=result_next_id.astype(ID_DTYPE),
        count=count,
        overflow=overflow.astype(MASK_DTYPE),
    )


def _allocate_counter(
    counters: IdCounters,
    request_mask: Array,
    field: str,
    constructor: Callable[[Array], IdCounters],
) -> tuple[IdCounters, IdAllocation]:
    allocation = allocate_ids(getattr(counters, field), request_mask)
    return constructor(allocation.next_id), allocation


def allocate_agent_ids(
    counters: IdCounters, request_mask: Array
) -> tuple[IdCounters, IdAllocation]:
    """Allocate agent IDs while preserving the other domains."""
    return _allocate_counter(
        counters,
        request_mask,
        "next_agent_id",
        lambda value: IdCounters(
            next_agent_id=value,
            next_genome_id=counters.next_genome_id,
            next_lineage_id=counters.next_lineage_id,
        ),
    )


def allocate_genome_ids(
    counters: IdCounters, request_mask: Array
) -> tuple[IdCounters, IdAllocation]:
    """Allocate genome IDs while preserving the other domains."""
    return _allocate_counter(
        counters,
        request_mask,
        "next_genome_id",
        lambda value: IdCounters(
            next_agent_id=counters.next_agent_id,
            next_genome_id=value,
            next_lineage_id=counters.next_lineage_id,
        ),
    )


def allocate_lineage_ids(
    counters: IdCounters, request_mask: Array
) -> tuple[IdCounters, IdAllocation]:
    """Allocate lineage IDs while preserving the other domains."""
    return _allocate_counter(
        counters,
        request_mask,
        "next_lineage_id",
        lambda value: IdCounters(
            next_agent_id=counters.next_agent_id,
            next_genome_id=counters.next_genome_id,
            next_lineage_id=value,
        ),
    )
