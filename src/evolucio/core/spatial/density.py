"""Derived global and local population density maps."""

# pyright: reportUnknownMemberType=false

import jax
import jax.numpy as jnp

from evolucio.core.dtypes import COUNT_DTYPE, REAL_DTYPE
from evolucio.core.types import Array


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"{name} must be a Python int, not bool")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_radius(radius: int) -> None:
    if isinstance(radius, bool) or not isinstance(radius, int):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("radius must be a Python int, not bool")
    if radius < 0:
        raise ValueError("radius must be non-negative")


def occupancy_to_density(occupancy: Array, *, max_agents: int) -> Array:
    """Normalize cell occupancy by the fixed population capacity."""
    _validate_positive_int(max_agents, "max_agents")
    return occupancy.astype(REAL_DTYPE) / jnp.asarray(max_agents, dtype=REAL_DTYPE)


def compute_local_population_count(occupancy: Array, *, radius: int) -> Array:
    """Sum a zero-padded square Chebyshev neighbourhood around every cell."""
    _validate_radius(radius)
    window_size = 2 * radius + 1
    return jax.lax.reduce_window(
        occupancy.astype(COUNT_DTYPE),
        jnp.asarray(0, dtype=COUNT_DTYPE),
        jax.lax.add,
        (window_size, window_size),
        (1, 1),
        "SAME",
    )


def compute_local_population_density(occupancy: Array, *, radius: int, max_agents: int) -> Array:
    """Normalize local agent counts by fixed population capacity, not window area."""
    local_count = compute_local_population_count(occupancy, radius=radius)
    return occupancy_to_density(local_count, max_agents=max_agents)
