"""Vectorised construction of fixed-shape world occupancy maps."""

# pyright: reportUnknownMemberType=false

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.dtypes import COUNT_DTYPE, INDEX_DTYPE
from evolucio.core.state import PopulationState, WorldState
from evolucio.core.types import Array
from evolucio.core.world.bounds import positions_in_bounds


class OccupancyResult(eqx.Module):
    """Occupancy map and count of alive agents outside the world."""

    occupancy: Array
    invalid_alive_count: Array


class WorldOccupancyUpdate(eqx.Module):
    """Immutable world update and its spatial invariant diagnostic."""

    world: WorldState
    invalid_alive_count: Array


def _validate_dimension(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"{name} must be a Python int, not bool")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def compute_occupancy(population: PopulationState, *, width: int, height: int) -> OccupancyResult:
    """Count valid alive ``[x, y]`` positions in a ``[y, x]`` map."""
    _validate_dimension(width, "width")
    _validate_dimension(height, "height")
    in_bounds = positions_in_bounds(population.position, width=width, height=height)
    valid_alive = population.alive & in_bounds
    invalid_alive = population.alive & ~in_bounds
    safe_x = jnp.where(valid_alive, population.position[:, 0], 0).astype(INDEX_DTYPE)
    safe_y = jnp.where(valid_alive, population.position[:, 1], 0).astype(INDEX_DTYPE)
    flat_index = (safe_y * width + safe_x).astype(INDEX_DTYPE)
    weights = valid_alive.astype(COUNT_DTYPE)
    occupancy = (
        jnp.zeros((height * width,), dtype=COUNT_DTYPE)
        .at[flat_index]
        .add(weights)
        .reshape((height, width))
    )
    return OccupancyResult(
        occupancy=occupancy,
        invalid_alive_count=jnp.sum(invalid_alive, dtype=COUNT_DTYPE),
    )


def rebuild_world_occupancy(
    world: WorldState,
    population: PopulationState,
    *,
    width: int,
    height: int,
) -> WorldOccupancyUpdate:
    """Return a world whose only replaced field is the occupancy map."""
    result = compute_occupancy(population, width=width, height=height)
    return WorldOccupancyUpdate(
        world=WorldState(
            resources=world.resources,
            environment=world.environment,
            occupancy=result.occupancy,
        ),
        invalid_alive_count=result.invalid_alive_count,
    )
