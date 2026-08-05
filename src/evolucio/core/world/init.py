"""Pure deterministic construction of fixed-shape world fields."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp

from evolucio.core.codes import RngStreamCode
from evolucio.core.dtypes import COUNT_DTYPE, REAL_DTYPE
from evolucio.core.rng import derive_indexed_key, derive_stream_key
from evolucio.core.state import WorldState
from evolucio.core.types import Array


class WorldInitializationConfig(Protocol):
    """Structural core-owned contract for world initialization inputs."""

    width: int
    height: int
    boundary_mode: str
    resource_distribution: str
    resource_capacity: Array
    initial_resource_mean: Array
    resource_patch_count: int
    resource_patch_radius: Array
    resource_patch_contrast: Array
    environment_initial_value: Array


def _initialize_patches(config: WorldInitializationConfig, key: Array) -> Array:
    x_key = derive_indexed_key(key, 0)
    y_key = derive_indexed_key(key, 1)
    center_x = jax.random.uniform(
        x_key,
        (config.resource_patch_count,),
        dtype=REAL_DTYPE,
        maxval=jnp.asarray(config.width, dtype=REAL_DTYPE),
    )
    center_y = jax.random.uniform(
        y_key,
        (config.resource_patch_count,),
        dtype=REAL_DTYPE,
        maxval=jnp.asarray(config.height, dtype=REAL_DTYPE),
    )
    grid_x = jnp.arange(config.width, dtype=REAL_DTYPE)[None, None, :]
    grid_y = jnp.arange(config.height, dtype=REAL_DTYPE)[None, :, None]
    distance_squared = (grid_x - center_x[:, None, None]) ** 2 + (
        grid_y - center_y[:, None, None]
    ) ** 2
    raw = jnp.sum(jnp.exp(-distance_squared / (2 * config.resource_patch_radius**2)), axis=0)
    centered = raw - jnp.mean(raw)
    epsilon = jnp.asarray(jnp.finfo(REAL_DTYPE).eps, dtype=REAL_DTYPE)
    pattern = centered / jnp.maximum(jnp.max(jnp.abs(centered)), epsilon)
    pattern = pattern - jnp.mean(pattern)
    margin = jnp.minimum(
        config.initial_resource_mean,
        config.resource_capacity - config.initial_resource_mean,
    )
    resources = config.initial_resource_mean + config.resource_patch_contrast * margin * pattern
    # Protect only against float32 rounding; host validation establishes the valid interval.
    return jnp.clip(resources, 0, config.resource_capacity).astype(REAL_DTYPE)


def initialize_resources(config: WorldInitializationConfig, key: Array) -> Array:
    """Construct the configured uniform or Gaussian-patch resource field."""
    shape = (config.height, config.width)
    if config.resource_distribution == "uniform":
        return jnp.full(shape, config.initial_resource_mean, dtype=REAL_DTYPE)
    if config.resource_distribution == "patches":
        return _initialize_patches(config, key)
    raise ValueError(f"unsupported resource distribution: {config.resource_distribution}")


def initialize_environment(config: WorldInitializationConfig) -> Array:
    """Construct the uniform basal environmental field."""
    return jnp.full(
        (config.height, config.width), config.environment_initial_value, dtype=REAL_DTYPE
    )


def initialize_world(config: WorldInitializationConfig, root_key: Array) -> WorldState:
    """Build a complete initial world without advancing persistent RNG state."""
    world_key = derive_stream_key(root_key, RngStreamCode.WORLD_INITIALIZATION)
    resource_key = derive_stream_key(world_key, RngStreamCode.RESOURCE_INITIALIZATION)
    return WorldState(
        resources=initialize_resources(config, resource_key),
        environment=initialize_environment(config),
        occupancy=jnp.zeros((config.height, config.width), dtype=COUNT_DTYPE),
    )
