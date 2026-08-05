"""Pure vectorized simultaneous cardinal movement resolution."""

# pyright: reportUnknownMemberType=false

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.codes import ActionCode
from evolucio.core.dtypes import CODE_DTYPE, COUNT_DTYPE, ID_DTYPE, INDEX_DTYPE
from evolucio.core.rng import derive_entity_keys
from evolucio.core.spatial import rebuild_world_occupancy
from evolucio.core.state import PopulationState, WorldState
from evolucio.core.types import Array
from evolucio.core.world.bounds import positions_in_bounds

from .contract import ActionValidationCode, action_deltas, is_movement_action
from .movement_schema import MovementConflictPriorityStreamCode, MovementResolutionCode
from .validate import ActionValidationResult

_UINT32_MAX = jnp.asarray(jnp.iinfo(jnp.uint32).max, dtype=jnp.uint32)


def _random_uint32(key: Array) -> Array:
    return jax.random.bits(key, shape=(), dtype=jnp.uint32)


def _position_field(state: PopulationState) -> Array:
    return state.position


class MovementResolutionResult(eqx.Module):
    """Updated state and fixed-shape diagnostics for the movement phase."""

    population: PopulationState
    world: WorldState
    actions_after_movement: Array
    movement_codes: Array
    contested_destination_count: Array
    unresolved_priority_collision_destination_count: Array
    invalid_movement_input_count: Array
    invalid_alive_position_count_after: Array


def _generate_movement_priorities(base_key: Array, agent_ids: Array) -> tuple[Array, Array, Array]:
    """Generate three independent uint32 priorities for every entity identity."""
    entity_keys = derive_entity_keys(base_key, agent_ids)

    def component(stream: MovementConflictPriorityStreamCode) -> Array:
        keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(entity_keys, int(stream))
        return jax.vmap(_random_uint32)(keys)

    return (
        component(MovementConflictPriorityStreamCode.PRIORITY_0),
        component(MovementConflictPriorityStreamCode.PRIORITY_1),
        component(MovementConflictPriorityStreamCode.PRIORITY_2),
    )


def _resolve_claims(
    eligible: Array,
    flat_targets: Array,
    priorities: tuple[Array, Array, Array],
    *,
    cell_count: int,
) -> tuple[Array, Array, Array, Array]:
    """Resolve destination claims lexicographically without positional fallback."""
    safe_targets = jnp.where(eligible, flat_targets, 0).astype(INDEX_DTYPE)
    claim_count = (
        jnp.zeros((cell_count,), dtype=COUNT_DTYPE)
        .at[safe_targets]
        .add(eligible.astype(COUNT_DTYPE))
    )
    finalists = eligible
    for priority in priorities:
        contributions = jnp.where(finalists, priority, _UINT32_MAX)
        minima = jax.ops.segment_min(contributions, safe_targets, num_segments=cell_count)
        finalists = finalists & (priority == minima[safe_targets])
    finalist_count = (
        jnp.zeros((cell_count,), dtype=COUNT_DTYPE)
        .at[safe_targets]
        .add(finalists.astype(COUNT_DTYPE))
    )
    unresolved_cells = finalist_count > 1
    winner = finalists & (finalist_count[safe_targets] == 1)
    priority_collision = eligible & unresolved_cells[safe_targets]
    return winner, priority_collision, claim_count, finalist_count


def resolve_cardinal_movement(
    *,
    population: PopulationState,
    world: WorldState,
    action_validation: ActionValidationResult,
    movement_conflict_key: Array,
    width: int,
    height: int,
) -> MovementResolutionResult:
    """Resolve routed cardinal movement against the phase-start occupancy snapshot."""
    routed = action_validation.routed_actions
    targets = action_validation.move_targets
    requested = is_movement_action(routed)
    current_in_bounds = positions_in_bounds(population.position, width=width, height=height)
    target_in_bounds = positions_in_bounds(targets, width=width, height=height)
    valid_id = population.agent_id >= jnp.asarray(0, dtype=ID_DTYPE)
    expected_targets = population.position + action_deltas(routed)
    matching_target = jnp.all(targets == expected_targets, axis=-1)
    valid_input = (
        population.alive
        & requested
        & (action_validation.validation_codes == ActionValidationCode.ACCEPTED)
        & valid_id
        & current_in_bounds
        & target_in_bounds
        & matching_target
    )
    invalid_input = requested & ~valid_input

    safe_targets_xy = jnp.where(target_in_bounds[..., None], targets, 0)
    flat_targets = (
        safe_targets_xy[:, 1] * jnp.asarray(width, dtype=INDEX_DTYPE) + safe_targets_xy[:, 0]
    ).astype(INDEX_DTYPE)
    destination_occupancy = world.occupancy.reshape(-1)[flat_targets]
    occupied = valid_input & (destination_occupancy != 0)
    eligible = valid_input & (destination_occupancy == 0)

    safe_ids = jnp.where(valid_id, population.agent_id, 0).astype(ID_DTYPE)
    priorities = _generate_movement_priorities(movement_conflict_key, safe_ids)
    winner, priority_collision, claim_count, finalist_count = _resolve_claims(
        eligible, flat_targets, priorities, cell_count=width * height
    )

    # Apply result precedence explicitly from lowest to highest priority.
    codes = jnp.full(routed.shape, MovementResolutionCode.CONFLICT_LOST, dtype=CODE_DTYPE)
    codes = jnp.where(winner, MovementResolutionCode.MOVED, codes)
    codes = jnp.where(priority_collision, MovementResolutionCode.PRIORITY_COLLISION, codes)
    codes = jnp.where(occupied, MovementResolutionCode.DESTINATION_OCCUPIED, codes)
    codes = jnp.where(invalid_input, MovementResolutionCode.INVALID_MOVEMENT_INPUT, codes)
    codes = jnp.where(~requested, MovementResolutionCode.NOT_MOVEMENT, codes).astype(CODE_DTYPE)

    stay = jnp.asarray(ActionCode.STAY, dtype=CODE_DTYPE)
    actions_after = jnp.where(requested & ~winner, stay, routed).astype(CODE_DTYPE)
    positions_after = jnp.where(winner[:, None], targets, population.position)
    population_after = eqx.tree_at(_position_field, population, positions_after)
    occupancy_update = rebuild_world_occupancy(world, population_after, width=width, height=height)
    return MovementResolutionResult(
        population=population_after,
        world=occupancy_update.world,
        actions_after_movement=actions_after,
        movement_codes=codes,
        contested_destination_count=jnp.sum(claim_count > 1, dtype=COUNT_DTYPE),
        unresolved_priority_collision_destination_count=jnp.sum(
            finalist_count > 1, dtype=COUNT_DTYPE
        ),
        invalid_movement_input_count=jnp.sum(invalid_input, dtype=COUNT_DTYPE),
        invalid_alive_position_count_after=occupancy_update.invalid_alive_count,
    )
