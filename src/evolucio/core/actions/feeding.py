"""Pure vectorized simultaneous local feeding resolution."""

# pyright: reportUnknownMemberType=false

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.codes import ActionCode
from evolucio.core.dtypes import CODE_DTYPE, COUNT_DTYPE, INDEX_DTYPE, REAL_DTYPE
from evolucio.core.spatial.gather import gather_map_values
from evolucio.core.state import PopulationState, WorldState
from evolucio.core.types import Array
from evolucio.core.world.bounds import positions_in_bounds

from .feeding_schema import FeedingResolutionCode


class FeedingResolutionResult(eqx.Module):
    """Updated state and fixed-shape diagnostics for the feeding phase."""

    population: PopulationState
    world: WorldState
    actions_after_feeding: Array
    feeding_codes: Array
    resource_demand: Array
    resource_consumed: Array
    energy_gained: Array
    contested_resource_cell_count: Array
    resource_limited_cell_count: Array
    invalid_feeding_input_count: Array


def _energy_field(state: PopulationState) -> Array:
    return state.energy


def _resources_field(state: WorldState) -> Array:
    return state.resources


def resolve_feeding(
    *,
    population: PopulationState,
    world: WorldState,
    actions_after_movement: Array,
    maximum_energy: Array,
    energy_gain_per_resource: Array,
    feeding_max_resource_intake: Array,
    width: int,
    height: int,
) -> FeedingResolutionResult:
    """Resolve feasible EAT demands proportionally within each current cell."""
    requested = actions_after_movement == jnp.asarray(ActionCode.EAT, dtype=CODE_DTYPE)
    in_bounds = positions_in_bounds(population.position, width=width, height=height)
    local_resource = gather_map_values(
        world.resources, population.position, width=width, height=height, fill_value=0
    )
    valid = (
        requested
        & population.alive
        & in_bounds
        & jnp.isfinite(population.energy)
        & (population.energy >= 0)
        & jnp.isfinite(local_resource)
        & (local_resource >= 0)
    )
    invalid = requested & ~valid
    has_resource = local_resource > 0
    headroom = jnp.maximum(maximum_energy - population.energy, 0)
    has_capacity = headroom > 0
    feasible = valid & has_resource & has_capacity
    needed = headroom / energy_gain_per_resource
    demand = jnp.where(feasible, jnp.minimum(feeding_max_resource_intake, needed), 0).astype(
        REAL_DTYPE
    )

    safe_positions = jnp.where(in_bounds[:, None], population.position, 0)
    flat_cell = (
        safe_positions[:, 1] * jnp.asarray(width, dtype=INDEX_DTYPE) + safe_positions[:, 0]
    ).astype(INDEX_DTYPE)
    cell_count = width * height
    consumers = demand > 0
    consumer_count = (
        jnp.zeros((cell_count,), dtype=COUNT_DTYPE).at[flat_cell].add(consumers.astype(COUNT_DTYPE))
    )
    total_demand = jnp.zeros((cell_count,), dtype=REAL_DTYPE).at[flat_cell].add(demand)
    available = world.resources.reshape(-1)
    safe_total = jnp.where(total_demand > 0, total_demand, 1)
    ratio = jnp.where(
        (total_demand > 0) & jnp.isfinite(available) & (available >= 0),
        jnp.minimum(1, available / safe_total),
        0,
    )
    consumed = jnp.where(consumers, demand * ratio[flat_cell], 0).astype(REAL_DTYPE)
    consumed_by_cell = jnp.zeros((cell_count,), dtype=REAL_DTYPE).at[flat_cell].add(consumed)
    # Correct any aggregate float32 overshoot with one common per-cell factor. Moving the
    # factor one representable value toward zero keeps the correction conservative while
    # preserving proportional neutrality between consumers.
    correction = jnp.where(
        consumed_by_cell > available,
        jnp.nextafter(available / consumed_by_cell, jnp.zeros_like(available)),
        1,
    )
    consumed = jnp.where(consumers, consumed * correction[flat_cell], 0).astype(REAL_DTYPE)
    consumed_by_cell = jnp.zeros((cell_count,), dtype=REAL_DTYPE).at[flat_cell].add(consumed)
    # Float32 scatter accumulation can round a proportional allocation a few ulps above
    # ``available``. Clamp the debit defensively so a valid cell never becomes negative.
    resources_after = jnp.maximum(available - consumed_by_cell, 0).reshape(world.resources.shape)

    theoretical_gain = consumed * energy_gain_per_resource
    energy_after = jnp.where(
        consumers,
        jnp.minimum(maximum_energy, population.energy + theoretical_gain),
        population.energy,
    )
    gained = jnp.where(consumers, energy_after - population.energy, 0).astype(REAL_DTYPE)

    full = consumers & (consumed >= demand)
    partial = consumers & ~full
    codes = jnp.full(requested.shape, FeedingResolutionCode.FED_FULL, dtype=CODE_DTYPE)
    codes = jnp.where(partial, FeedingResolutionCode.FED_PARTIAL, codes)
    codes = jnp.where(valid & ~has_capacity, FeedingResolutionCode.NO_ENERGY_CAPACITY, codes)
    codes = jnp.where(valid & ~has_resource, FeedingResolutionCode.NO_RESOURCE, codes)
    codes = jnp.where(invalid, FeedingResolutionCode.INVALID_FEEDING_INPUT, codes)
    codes = jnp.where(~requested, FeedingResolutionCode.NOT_FEEDING, codes).astype(CODE_DTYPE)
    failed = requested & ~(
        (codes == FeedingResolutionCode.FED_FULL) | (codes == FeedingResolutionCode.FED_PARTIAL)
    )
    actions_after = jnp.where(failed, ActionCode.STAY, actions_after_movement).astype(CODE_DTYPE)
    population_after = eqx.tree_at(_energy_field, population, energy_after)
    world_after = eqx.tree_at(_resources_field, world, resources_after)
    limited = total_demand > available
    return FeedingResolutionResult(
        population=population_after,
        world=world_after,
        actions_after_feeding=actions_after,
        feeding_codes=codes,
        resource_demand=demand,
        resource_consumed=consumed,
        energy_gained=gained,
        contested_resource_cell_count=jnp.sum((consumer_count > 1) & limited, dtype=COUNT_DTYPE),
        resource_limited_cell_count=jnp.sum(limited, dtype=COUNT_DTYPE),
        invalid_feeding_input_count=jnp.sum(invalid, dtype=COUNT_DTYPE),
    )
