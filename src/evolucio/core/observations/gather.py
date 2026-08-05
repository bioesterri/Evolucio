"""Vectorized spatial gathering for local observations."""

# pyright: reportUnknownMemberType=false

import jax.numpy as jnp

from evolucio.core.spatial.gather import gather_map_values as gather_map_values
from evolucio.core.types import Array


def build_cardinal_ray_positions(positions: Array, *, radius: int) -> Array:
    """Build ``[C, north/south/east/west, radius, xy]`` coordinates."""
    distance = jnp.arange(1, radius + 1, dtype=positions.dtype)
    directions = jnp.asarray(((0, -1), (0, 1), (1, 0), (-1, 0)), dtype=positions.dtype)
    offsets = directions[:, None, :] * distance[None, :, None]
    return positions[:, None, None, :] + offsets[None, :, :, :]
