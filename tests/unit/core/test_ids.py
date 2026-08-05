import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from evolucio.core import (
    COUNT_DTYPE,
    FIRST_ID,
    ID_DTYPE,
    MASK_DTYPE,
    MAX_NEXT_ID,
    NULL_ID,
    IdCounters,
    allocate_agent_ids,
    allocate_genome_ids,
    allocate_ids,
    allocate_lineage_ids,
    create_id_counters,
)


def mask(values: list[bool]) -> jax.Array:
    return jnp.asarray(values, dtype=MASK_DTYPE)


def scalar(value: int) -> jax.Array:
    return jnp.asarray(value, dtype=ID_DTYPE)


def test_counter_creation_contract_and_pytree() -> None:
    default = create_id_counters()
    explicit = create_id_counters(next_agent_id=1, next_genome_id=2, next_lineage_id=3)

    assert isinstance(default, IdCounters)
    assert [int(x) for x in jax.tree.leaves(default)] == [FIRST_ID] * 3
    assert [int(x) for x in jax.tree.leaves(explicit)] == [1, 2, 3]
    assert all(x.shape == () and x.dtype == jnp.dtype(ID_DTYPE) for x in jax.tree.leaves(default))
    leaves, tree = jax.tree.flatten(default)
    assert isinstance(jax.tree.unflatten(tree, leaves), IdCounters)


@pytest.mark.parametrize("value", [-1, MAX_NEXT_ID + 1])
def test_counter_creation_rejects_out_of_range(value: int) -> None:
    with pytest.raises(ValueError):
        create_id_counters(next_agent_id=value)


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_counter_creation_rejects_non_integer(value: object) -> None:
    with pytest.raises(TypeError):
        create_id_counters(next_genome_id=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("start", "requests", "values", "next_value", "count"),
    [
        (5, [False, False, False], [-1, -1, -1], 5, 0),
        (5, [False, True, False], [-1, 5, -1], 6, 1),
        (10, [False, True, False, True, True], [-1, 10, -1, 11, 12], 13, 3),
    ],
)
def test_allocate_ids_cases(
    start: int, requests: list[bool], values: list[int], next_value: int, count: int
) -> None:
    result = allocate_ids(scalar(start), mask(requests))
    assert result.values.tolist() == values
    assert int(result.next_id) == next_value
    assert int(result.count) == count
    assert not bool(result.overflow)
    assert result.values.dtype == jnp.dtype(ID_DTYPE)
    assert result.count.dtype == jnp.dtype(COUNT_DTYPE)
    assert result.overflow.dtype == jnp.dtype(MASK_DTYPE)


def test_consecutive_allocations_never_reuse_identifiers() -> None:
    first = allocate_ids(scalar(10), mask([True, False, True]))
    second = allocate_ids(first.next_id, mask([True, True, False]))
    assert first.values[first.values != NULL_ID].tolist() == [10, 11]
    assert second.values[second.values != NULL_ID].tolist() == [12, 13]


def test_overflow_is_atomic_without_wraparound() -> None:
    result = allocate_ids(scalar(MAX_NEXT_ID - 1), mask([True, True, False]))
    assert bool(result.overflow)
    assert result.values.tolist() == [NULL_ID, NULL_ID, NULL_ID]
    assert int(result.next_id) == MAX_NEXT_ID - 1
    assert int(result.count) == 2


def test_eager_and_jit_allocations_are_equivalent_and_fixed_shape() -> None:
    eager = allocate_ids(scalar(4), mask([True, False, True, False]))
    compiled = jax.jit(allocate_ids)(scalar(4), mask([True, False, True, False]))
    empty = allocate_ids(scalar(4), mask([False, False, False, False]))
    assert jax.tree.all(jax.tree.map(jnp.array_equal, eager, compiled))
    assert eager.values.shape == empty.values.shape == (4,)


@pytest.mark.parametrize(
    ("allocator", "field"),
    [
        (allocate_agent_ids, "next_agent_id"),
        (allocate_genome_ids, "next_genome_id"),
        (allocate_lineage_ids, "next_lineage_id"),
    ],
)
def test_specialized_allocators_only_advance_their_counter(allocator: object, field: str) -> None:
    counters = create_id_counters(next_agent_id=10, next_genome_id=20, next_lineage_id=30)
    before = [int(value) for value in jax.tree.leaves(counters)]
    updated, allocation = allocator(counters, mask([False, True, True]))  # type: ignore[operator]
    compiled_updated, compiled_allocation = eqx.filter_jit(allocator)(  # type: ignore[arg-type]
        counters, mask([False, True, True])
    )

    start = getattr(counters, field).item()
    assert allocation.values.tolist() == [NULL_ID, start, start + 1]
    assert int(getattr(updated, field)) == int(getattr(counters, field)) + 2
    for other in {"next_agent_id", "next_genome_id", "next_lineage_id"} - {field}:
        assert int(getattr(updated, other)) == int(getattr(counters, other))
    assert [int(value) for value in jax.tree.leaves(counters)] == before
    assert jax.tree.all(jax.tree.map(jnp.array_equal, updated, compiled_updated))
    assert jax.tree.all(jax.tree.map(jnp.array_equal, allocation, compiled_allocation))


def test_specialized_overflow_preserves_all_counters() -> None:
    counters = create_id_counters(next_agent_id=MAX_NEXT_ID - 1)
    updated, result = allocate_agent_ids(counters, mask([True, True]))
    assert bool(result.overflow)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, counters, updated))
