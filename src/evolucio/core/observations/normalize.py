"""Numerically safe observation normalization helpers."""

# pyright: reportUnknownMemberType=false

import jax.numpy as jnp

from evolucio.core.dtypes import REAL_DTYPE
from evolucio.core.types import Array


def normalize_positive(value: Array, scale: Array) -> Array:
    """Normalize a positive magnitude, returning zero for a non-positive scale."""
    value = jnp.asarray(value, dtype=REAL_DTYPE)
    scale = jnp.asarray(scale, dtype=REAL_DTYPE)
    safe_scale = jnp.where(scale > 0, scale, jnp.ones_like(scale))
    return jnp.where(scale > 0, jnp.clip(value / safe_scale, 0.0, 1.0), 0.0).astype(REAL_DTYPE)


def normalize_signed_margin(value: Array, threshold: Array, scale: Array) -> Array:
    """Normalize a signed threshold margin, returning zero for an invalid scale."""
    value = jnp.asarray(value, dtype=REAL_DTYPE)
    threshold = jnp.asarray(threshold, dtype=REAL_DTYPE)
    scale = jnp.asarray(scale, dtype=REAL_DTYPE)
    safe_scale = jnp.where(scale > 0, scale, jnp.ones_like(scale))
    return jnp.where(scale > 0, jnp.clip((value - threshold) / safe_scale, -1.0, 1.0), 0.0).astype(
        REAL_DTYPE
    )
