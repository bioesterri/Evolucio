"""Construction of fixed-size local observations."""

# pyright: reportUnknownMemberType=false

from typing import TYPE_CHECKING

import jax.numpy as jnp

from evolucio.core.dtypes import REAL_DTYPE
from evolucio.core.spatial import compute_local_population_count
from evolucio.core.state import SimulationState
from evolucio.core.types import Array
from evolucio.core.world.bounds import positions_in_bounds

from .gather import build_cardinal_ray_positions, gather_map_values
from .normalize import normalize_positive, normalize_signed_margin

if TYPE_CHECKING:
    from evolucio.config.compile import CoreConfig


def build_observations(state: SimulationState, config: "CoreConfig") -> Array:
    """Build pure local observations for every fixed-capacity population slot."""
    world = state.world
    population = state.population
    width, height = config.world.width, config.world.height
    radius = config.observations.perception_radius
    positions = population.position
    valid = population.alive & positions_in_bounds(positions, width=width, height=height)

    current_resource = gather_map_values(
        world.resources, positions, width=width, height=height, fill_value=0.0
    )
    environment = gather_map_values(
        world.environment, positions, width=width, height=height, fill_value=0.0
    )
    ray_positions = build_cardinal_ray_positions(positions, radius=radius)
    ray_resources = gather_map_values(
        world.resources, ray_positions, width=width, height=height, fill_value=0.0
    )
    ray_occupancy = gather_map_values(
        world.occupancy, ray_positions, width=width, height=height, fill_value=0
    )
    resource_directions = normalize_positive(
        jnp.sum(ray_resources, axis=-1), config.world.resource_capacity * radius
    )
    agent_directions = normalize_positive(
        jnp.sum(ray_occupancy, axis=-1), jnp.asarray(config.population.max_agents, dtype=REAL_DTYPE)
    )

    local_counts = compute_local_population_count(world.occupancy, radius=radius)
    focal_count = gather_map_values(
        local_counts, positions, width=width, height=height, fill_value=0
    )
    other_count = jnp.maximum(focal_count - jnp.asarray(1, dtype=focal_count.dtype), 0)
    local_density = normalize_positive(
        other_count, jnp.asarray(config.population.max_agents, dtype=REAL_DTYPE)
    )

    x, y = positions[:, 0], positions[:, 1]
    blocked = (
        (y <= 0).astype(REAL_DTYPE)
        + (y >= height - 1).astype(REAL_DTYPE) * 2
        + (x >= width - 1).astype(REAL_DTYPE) * 4
        + (x <= 0).astype(REAL_DTYPE) * 8
    ) / 15.0
    observations = jnp.stack(
        (
            normalize_positive(population.energy, config.energy.max_energy),
            normalize_positive(population.age, config.evolution.max_age),
            normalize_signed_margin(
                population.energy, config.energy.reproduction_threshold, config.energy.max_energy
            ),
            normalize_positive(current_resource, config.world.resource_capacity),
            resource_directions[:, 0],
            resource_directions[:, 1],
            resource_directions[:, 2],
            resource_directions[:, 3],
            agent_directions[:, 0],
            agent_directions[:, 1],
            agent_directions[:, 2],
            agent_directions[:, 3],
            local_density,
            jnp.clip(environment, 0.0, 1.0),
            blocked,
        ),
        axis=1,
    ).astype(REAL_DTYPE)
    return jnp.where(valid[:, None], observations, jnp.zeros_like(observations))
