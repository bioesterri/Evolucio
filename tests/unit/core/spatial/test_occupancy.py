"""Tests for vectorised occupancy construction and world updates."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evolucio.core.dtypes import COUNT_DTYPE, INDEX_DTYPE, MASK_DTYPE, REAL_DTYPE
from evolucio.core.population import create_empty_population
from evolucio.core.spatial import (
    OccupancyResult,
    WorldOccupancyUpdate,
    compute_occupancy,
    rebuild_world_occupancy,
)
from evolucio.core.state import PopulationState, WorldState


def population_with(positions: list[list[int]], alive: list[bool]) -> PopulationState:
    population = create_empty_population(len(alive))
    return eqx.tree_at(
        lambda state: (state.position, state.alive),
        population,
        (jnp.asarray(positions, dtype=INDEX_DTYPE), jnp.asarray(alive, dtype=MASK_DTYPE)),
    )


def test_empty_population_has_fixed_zero_map() -> None:
    result = compute_occupancy(create_empty_population(4), width=5, height=4)
    assert isinstance(result, OccupancyResult)
    assert result.occupancy.shape == (4, 5)
    assert result.occupancy.dtype == COUNT_DTYPE
    assert int(result.occupancy.sum()) == 0
    assert int(result.invalid_alive_count) == 0


def test_coordinates_are_x_y_but_map_is_y_x() -> None:
    population = population_with([[3, 2], [-1, -1]], [True, False])
    result = compute_occupancy(population, width=5, height=4)
    assert int(result.occupancy[2, 3]) == 1
    assert int(result.occupancy.sum()) == 1


def test_duplicate_positions_are_accumulated() -> None:
    population = population_with([[0, 0]] * 4, [True] * 4)
    result = compute_occupancy(population, width=1, height=1)
    assert result.occupancy.dtype == COUNT_DTYPE
    assert int(result.occupancy[0, 0]) == 4


def test_inactive_positions_never_contribute() -> None:
    population = population_with([[-1, -1], [1, 1], [99, 99]], [False, False, False])
    result = compute_occupancy(population, width=2, height=2)
    assert int(result.occupancy.sum()) == 0
    assert int(result.invalid_alive_count) == 0


def test_invalid_alive_positions_are_reported_without_indexing() -> None:
    population = population_with([[-1, 0], [0, -1], [3, 0], [0, 2], [0, 0]], [True] * 5)
    eager = compute_occupancy(population, width=3, height=2)
    compiled = eqx.filter_jit(compute_occupancy)(population, width=3, height=2)
    assert int(eager.occupancy[0, 0]) == 1
    assert int(eager.occupancy.sum()) == 1
    assert int(eager.invalid_alive_count) == 4
    assert int(eager.occupancy.sum() + eager.invalid_alive_count) == int(population.alive.sum())
    assert jnp.array_equal(compiled.occupancy, eager.occupancy)


def test_world_rebuild_preserves_non_occupancy_fields_and_is_jittable() -> None:
    world = WorldState(
        resources=jnp.arange(6, dtype=REAL_DTYPE).reshape((2, 3)),
        environment=jnp.asarray([2, 7], dtype=INDEX_DTYPE),
        occupancy=jnp.full((2, 3), 9, dtype=COUNT_DTYPE),
    )
    population = population_with([[1, 0], [1, 0]], [True, True])
    eager = rebuild_world_occupancy(world, population, width=3, height=2)
    update = eqx.filter_jit(rebuild_world_occupancy)(world, population, width=3, height=2)
    assert isinstance(update, WorldOccupancyUpdate)
    assert eager.world.resources is world.resources
    assert eager.world.environment is world.environment
    assert jnp.array_equal(update.world.resources, world.resources)
    assert jnp.array_equal(update.world.environment, world.environment)
    assert int(update.world.occupancy[0, 1]) == 2
    assert int(world.occupancy[0, 1]) == 9
    assert int(update.invalid_alive_count) == 0


def test_rebuild_world_occupancy_is_scan_compatible() -> None:
    world = WorldState(
        resources=jnp.zeros((2, 2), dtype=REAL_DTYPE),
        environment=jnp.zeros((1,), dtype=INDEX_DTYPE),
        occupancy=jnp.zeros((2, 2), dtype=COUNT_DTYPE),
    )
    population = population_with([[0, 0], [1, 1]], [True, True])
    positions = jnp.asarray([[[0, 0], [1, 1]], [[1, 0], [0, 1]]], dtype=INDEX_DTYPE)

    def body(carry: WorldState, position: jax.Array) -> tuple[WorldState, jax.Array]:
        current = eqx.tree_at(lambda state: state.position, population, position)
        updated = rebuild_world_occupancy(carry, current, width=2, height=2).world
        return updated, updated.occupancy

    final_world, maps = jax.jit(lambda: jax.lax.scan(body, world, positions))()
    assert maps.shape == (2, 2, 2)
    assert jnp.array_equal(final_world.occupancy, maps[-1])


@given(
    positions=st.lists(st.tuples(st.integers(-1, 3), st.integers(-1, 2)), min_size=6, max_size=6),
    alive=st.lists(st.booleans(), min_size=6, max_size=6),
)
@settings(max_examples=25, deadline=None)
def test_occupancy_balance_determinism_and_slot_permutation(
    positions: list[tuple[int, int]], alive: list[bool]
) -> None:
    coordinates = [list(position) for position in positions]
    population = population_with(coordinates, alive)
    result = compute_occupancy(population, width=3, height=2)
    permutation = jnp.asarray([5, 2, 0, 4, 1, 3], dtype=INDEX_DTYPE)
    permuted = eqx.tree_at(
        lambda state: (state.position, state.alive),
        population,
        (population.position[permutation], population.alive[permutation]),
    )
    permuted_result = compute_occupancy(permuted, width=3, height=2)
    assert result.occupancy.shape == (2, 3)
    assert result.occupancy.dtype == COUNT_DTYPE
    assert bool(jnp.all(result.occupancy >= 0))
    assert int(result.occupancy.sum() + result.invalid_alive_count) == sum(alive)
    assert jnp.array_equal(result.occupancy, permuted_result.occupancy)
    assert jnp.array_equal(
        result.occupancy, compute_occupancy(population, width=3, height=2).occupancy
    )


@pytest.mark.parametrize("width,height", [(0, 1), (1, 0), (-1, 1)])
def test_world_dimensions_must_be_positive(width: int, height: int) -> None:
    with pytest.raises(ValueError):
        compute_occupancy(create_empty_population(1), width=width, height=height)


def test_new_modules_avoid_prohibited_patterns() -> None:
    root = Path(__file__).parents[4] / "src/evolucio/core/spatial"
    source = "\n".join(path.read_text() for path in root.glob("*.py"))
    for pattern in (
        "import numpy",
        " np.",
        "random.",
        "jax.random",
        "list[Agent]",
        "for agent in",
        "for y in",
        "for x in",
    ):
        assert pattern not in source
