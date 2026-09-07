"""Pure post-action energy-cost accounting."""

# pyright: reportUnknownMemberType=false

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.actions import FeedingResolutionCode, MovementResolutionCode, is_movement_action
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
    inconsistent = (
        ((~population.alive) & (moved | fed))
        | (moved & fed)
        | (moved & ~is_movement_action(actions_after_feeding))
        | (fed & (actions_after_feeding != jnp.asarray(ActionCode.EAT, dtype=CODE_DTYPE)))
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
