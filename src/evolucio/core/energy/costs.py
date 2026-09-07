"""Pure post-action energy-cost accounting."""

# pyright: reportUnknownMemberType=false

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.actions import (
    FEEDING_RESOLUTION_CODE_COUNT,
    MOVEMENT_RESOLUTION_CODE_COUNT,
    FeedingResolutionCode,
    MovementResolutionCode,
    is_movement_action,
)
from evolucio.core.codes import ActionCode
from evolucio.core.dtypes import CODE_DTYPE, COUNT_DTYPE, REAL_DTYPE
from evolucio.core.state import PopulationState
from evolucio.core.types import Array


class ActionEnergyCostResult(eqx.Module):
    """Population and per-slot costs from successful action outcomes."""

    population: PopulationState
    movement_cost_applied: Array
    feeding_cost_applied: Array
    action_cost_applied: Array
    invalid_action_cost_input_count: Array


def _energy_field(state: PopulationState) -> Array:
    return state.energy


def apply_action_energy_costs(
    *,
    population: PopulationState,
    actions_after_feeding: Array,
    movement_codes: Array,
    feeding_codes: Array,
    movement_energy_cost: Array,
    feeding_energy_cost: Array,
) -> ActionEnergyCostResult:
    """Debit only coherent, successful movement and feeding outcomes."""
    moved = movement_codes == jnp.asarray(MovementResolutionCode.MOVED, dtype=CODE_DTYPE)
    fed = (feeding_codes == jnp.asarray(FeedingResolutionCode.FED_FULL, dtype=CODE_DTYPE)) | (
        feeding_codes == jnp.asarray(FeedingResolutionCode.FED_PARTIAL, dtype=CODE_DTYPE)
    )
    movement_not_applicable = movement_codes == jnp.asarray(
        MovementResolutionCode.NOT_MOVEMENT, dtype=CODE_DTYPE
    )
    movement_failed = (
        (movement_codes == MovementResolutionCode.DESTINATION_OCCUPIED)
        | (movement_codes == MovementResolutionCode.CONFLICT_LOST)
        | (movement_codes == MovementResolutionCode.PRIORITY_COLLISION)
        | (movement_codes == MovementResolutionCode.INVALID_MOVEMENT_INPUT)
    )
    feeding_not_applicable = feeding_codes == jnp.asarray(
        FeedingResolutionCode.NOT_FEEDING, dtype=CODE_DTYPE
    )
    feeding_failed = (
        (feeding_codes == FeedingResolutionCode.NO_RESOURCE)
        | (feeding_codes == FeedingResolutionCode.NO_ENERGY_CAPACITY)
        | (feeding_codes == FeedingResolutionCode.INVALID_FEEDING_INPUT)
    )
    valid_movement_code = (movement_codes >= 0) & (movement_codes < MOVEMENT_RESOLUTION_CODE_COUNT)
    valid_feeding_code = (feeding_codes >= 0) & (feeding_codes < FEEDING_RESOLUTION_CODE_COUNT)
    movement_action = is_movement_action(actions_after_feeding)
    eat_action = actions_after_feeding == jnp.asarray(ActionCode.EAT, dtype=CODE_DTYPE)
    stay_action = actions_after_feeding == jnp.asarray(ActionCode.STAY, dtype=CODE_DTYPE)
    reproduce_action = actions_after_feeding == jnp.asarray(ActionCode.REPRODUCE, dtype=CODE_DTYPE)
    coherent_active_outcome = (
        (movement_action & moved & feeding_not_applicable)
        | (eat_action & movement_not_applicable & fed)
        | (reproduce_action & movement_not_applicable & feeding_not_applicable)
        | (
            stay_action
            & (
                (movement_not_applicable & (feeding_not_applicable | feeding_failed))
                | (movement_failed & feeding_not_applicable)
            )
        )
    )
    inconsistent = (
        ~valid_movement_code
        | ~valid_feeding_code
        | (population.alive & ~coherent_active_outcome)
        | ((~population.alive) & (moved | fed))
        | (population.alive & ~jnp.isfinite(population.energy))
    )
    applicable = population.alive & ~inconsistent
    zero = jnp.asarray(0, dtype=REAL_DTYPE)
    movement_cost = jnp.where(applicable & moved, movement_energy_cost, zero).astype(REAL_DTYPE)
    feeding_cost = jnp.where(applicable & fed, feeding_energy_cost, zero).astype(REAL_DTYPE)
    action_cost = (movement_cost + feeding_cost).astype(REAL_DTYPE)
    energy_after = jnp.where(applicable, population.energy - action_cost, population.energy)
    return ActionEnergyCostResult(
        population=eqx.tree_at(_energy_field, population, energy_after),
        movement_cost_applied=movement_cost,
        feeding_cost_applied=feeding_cost,
        action_cost_applied=action_cost,
        invalid_action_cost_input_count=jnp.sum(inconsistent, dtype=COUNT_DTYPE),
    )
