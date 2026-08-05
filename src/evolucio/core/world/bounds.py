"""Vectorised checks for closed, non-toroidal world boundaries."""

# pyright: reportUnknownMemberType=false

import jax.numpy as jnp

from evolucio.core.dtypes import MASK_DTYPE
from evolucio.core.types import Array


def positions_in_bounds(positions: Array, *, width: int, height: int) -> Array:
    """Return whether each ``[x, y]`` coordinate lies inside the world."""
    x = positions[..., 0]
    y = positions[..., 1]
    return jnp.asarray((x >= 0) & (x < width) & (y >= 0) & (y < height), dtype=MASK_DTYPE)
