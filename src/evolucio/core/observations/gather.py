"""Vectorized spatial gathering for local observations."""

# pyright: reportUnknownMemberType=false

import jax.numpy as jnp

from evolucio.core.types import Array
from evolucio.core.world.bounds import positions_in_bounds


def gather_map_values(
    values: Array, positions: Array, *, width: int, height: int, fill_value: float | int
) -> Array:
    """Gather ``values[y, x]`` and fill coordinates outside closed bounds."""
    in_bounds = positions_in_bounds(positions, width=width, height=height)
    safe_positions = jnp.where(in_bounds[..., None], positions, 0)
    linear = safe_positions[..., 1] * width + safe_positions[..., 0]
    gathered = values.reshape(-1)[linear]
    return jnp.where(in_bounds, gathered, jnp.asarray(fill_value, dtype=values.dtype))


def build_cardinal_ray_positions(positions: Array, *, radius: int) -> Array:
    """Build ``[C, north/south/east/west, radius, xy]`` coordinates."""
    distance = jnp.arange(1, radius + 1, dtype=positions.dtype)
    directions = jnp.asarray(((0, -1), (0, 1), (1, 0), (-1, 0)), dtype=positions.dtype)
    offsets = directions[:, None, :] * distance[None, :, None]
    return positions[:, None, None, :] + offsets[None, :, :, :]
