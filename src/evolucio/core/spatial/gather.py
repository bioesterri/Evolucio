"""Safe vectorized extraction from maps indexed by ``[x, y]`` positions."""

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
