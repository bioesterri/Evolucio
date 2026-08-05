"""Tests for derived global and local density maps."""

import equinox as eqx
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evolucio.core.dtypes import COUNT_DTYPE, REAL_DTYPE
from evolucio.core.world import (
    compute_local_population_count,
    compute_local_population_density,
    occupancy_to_density,
)


def test_cell_density_uses_fixed_capacity_and_preserves_shape() -> None:
    occupancy = jnp.asarray([[0, 1], [5, 10]], dtype=COUNT_DTYPE)
    original = occupancy.copy()
    density = occupancy_to_density(occupancy, max_agents=10)
    assert density.shape == occupancy.shape
    assert density.dtype == REAL_DTYPE
    assert jnp.allclose(density, jnp.asarray([[0.0, 0.1], [0.5, 1.0]]))
    assert jnp.array_equal(occupancy, original)
    compiled = eqx.filter_jit(occupancy_to_density)(occupancy, max_agents=10)
    assert jnp.array_equal(compiled, density)


def test_radius_zero_is_identity_for_count_and_density() -> None:
    occupancy = jnp.asarray([[0, 2], [1, 3]], dtype=COUNT_DTYPE)
    assert jnp.array_equal(compute_local_population_count(occupancy, radius=0), occupancy)
    assert jnp.array_equal(
        compute_local_population_density(occupancy, radius=0, max_agents=6),
        occupancy_to_density(occupancy, max_agents=6),
    )


def test_manual_zero_padded_local_count() -> None:
    occupancy = jnp.asarray([[0, 1, 0], [2, 0, 0], [0, 0, 3]], dtype=COUNT_DTYPE)
    expected = jnp.asarray([[3, 3, 1], [3, 6, 4], [2, 5, 3]], dtype=COUNT_DTYPE)
    actual = compute_local_population_count(occupancy, radius=1)
    assert jnp.array_equal(actual, expected)


def test_edges_do_not_wrap_and_focal_cell_is_included() -> None:
    occupancy = jnp.asarray([[1, 0, 4], [0, 0, 0], [2, 0, 3]], dtype=COUNT_DTYPE)
    local = compute_local_population_count(occupancy, radius=1)
    assert int(local[0, 0]) == 1
    assert int(local[2, 0]) == 2
    assert int(local[0, 2]) == 4
    assert int(local[2, 2]) == 3


def test_local_density_matches_count_over_capacity_under_jit() -> None:
    occupancy = jnp.asarray([[1, 2], [3, 4]], dtype=COUNT_DTYPE)
    eager = compute_local_population_density(occupancy, radius=1, max_agents=10)
    compiled = eqx.filter_jit(compute_local_population_density)(occupancy, radius=1, max_agents=10)
    expected = compute_local_population_count(occupancy, radius=1).astype(REAL_DTYPE) / 10
    assert eager.dtype == REAL_DTYPE
    assert jnp.array_equal(eager, expected)
    assert jnp.array_equal(compiled, eager)


@given(
    values=st.lists(st.integers(min_value=0, max_value=5), min_size=9, max_size=9),
    radius=st.integers(min_value=0, max_value=2),
)
@settings(max_examples=20, deadline=None)
def test_local_count_properties(values: list[int], radius: int) -> None:
    occupancy = jnp.asarray(values, dtype=COUNT_DTYPE).reshape((3, 3))
    local = compute_local_population_count(occupancy, radius=radius)
    assert local.shape == occupancy.shape
    assert local.dtype == COUNT_DTYPE
    assert bool(jnp.all(local >= occupancy))
    assert jnp.array_equal(local, compute_local_population_count(occupancy, radius=radius))


@pytest.mark.parametrize("radius", [-1, 0.5, True])
def test_radius_validation(radius: object) -> None:
    occupancy = jnp.zeros((1, 1), dtype=COUNT_DTYPE)
    with pytest.raises((TypeError, ValueError)):
        compute_local_population_count(occupancy, radius=radius)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_agents", [0, -1, True])
def test_capacity_validation(max_agents: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        occupancy_to_density(jnp.zeros((1, 1), dtype=COUNT_DTYPE), max_agents=max_agents)
