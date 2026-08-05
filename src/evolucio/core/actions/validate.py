"""Pure vectorized local validation and routing of action proposals."""

# pyright: reportUnknownMemberType=false

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.codes import ActionCode
from evolucio.core.dtypes import CODE_DTYPE
from evolucio.core.spatial import gather_map_values
from evolucio.core.types import Array
from evolucio.core.world.bounds import positions_in_bounds

from .contract import (
    ActionValidationCode,
    action_deltas,
    is_movement_action,
    is_valid_action_code,
)


class ActionValidationResult(eqx.Module):
    """Original proposals and their locally validated routing information."""

    proposed_actions: Array
    routed_actions: Array
    validation_codes: Array
    move_targets: Array


def validate_and_route_actions(
    *,
    proposed_actions: Array,
    alive: Array,
    positions: Array,
    resources: Array,
    width: int,
    height: int,
) -> ActionValidationResult:
    """Route locally valid proposals and replace every rejection with ``STAY``."""
    valid_code = is_valid_action_code(proposed_actions)
    stay = jnp.asarray(ActionCode.STAY, dtype=CODE_DTYPE)
    safe_actions = jnp.where(valid_code, proposed_actions, stay)
    actor_in_bounds = positions_in_bounds(positions, width=width, height=height)
    candidate_targets = positions + action_deltas(safe_actions)
    movement = is_movement_action(safe_actions)
    target_in_bounds = positions_in_bounds(candidate_targets, width=width, height=height)
    move_out_of_bounds = movement & ~target_in_bounds
    resource_at_position = gather_map_values(
        resources, positions, width=width, height=height, fill_value=0
    )
    eat_without_resource = (safe_actions == ActionCode.EAT) & (resource_at_position <= 0)

    # Build precedence explicitly from lowest to highest priority.
    accepted = jnp.asarray(ActionValidationCode.ACCEPTED, dtype=CODE_DTYPE)
    validation_codes = jnp.full(proposed_actions.shape, accepted, dtype=CODE_DTYPE)
    validation_codes = jnp.where(
        eat_without_resource,
        jnp.asarray(ActionValidationCode.EAT_NO_RESOURCE, dtype=CODE_DTYPE),
        validation_codes,
    )
    validation_codes = jnp.where(
        move_out_of_bounds,
        jnp.asarray(ActionValidationCode.MOVE_OUT_OF_BOUNDS, dtype=CODE_DTYPE),
        validation_codes,
    )
    validation_codes = jnp.where(
        ~actor_in_bounds,
        jnp.asarray(ActionValidationCode.INVALID_ACTOR_POSITION, dtype=CODE_DTYPE),
        validation_codes,
    )
    validation_codes = jnp.where(
        ~valid_code,
        jnp.asarray(ActionValidationCode.INVALID_ACTION_CODE, dtype=CODE_DTYPE),
        validation_codes,
    )
    validation_codes = jnp.where(
        ~alive,
        jnp.asarray(ActionValidationCode.INACTIVE_SLOT, dtype=CODE_DTYPE),
        validation_codes,
    )

    is_accepted = validation_codes == ActionValidationCode.ACCEPTED
    routed_actions = jnp.where(is_accepted, safe_actions, stay).astype(CODE_DTYPE)
    routed_movement = is_movement_action(routed_actions)
    move_targets = jnp.where(routed_movement[..., None], candidate_targets, positions)
    return ActionValidationResult(
        proposed_actions=proposed_actions,
        routed_actions=routed_actions,
        validation_codes=validation_codes,
        move_targets=move_targets,
    )
