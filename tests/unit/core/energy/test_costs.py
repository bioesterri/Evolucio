import equinox as eqx
import jax.numpy as jnp

from evolucio.core.actions import FeedingResolutionCode, MovementResolutionCode
from evolucio.core.codes import ActionCode
from evolucio.core.energy import apply_action_energy_costs
from evolucio.core.energy.costs import ActionEnergyCostResult
from evolucio.core.state import PopulationState

from .conftest import population


def apply(
    actions: list[int],
    movement: list[int],
    feeding: list[int],
    state: PopulationState | None = None,
) -> ActionEnergyCostResult:
    state = state or population([10] * len(actions), [True] * len(actions))
    return apply_action_energy_costs(
        population=state,
        actions_after_feeding=jnp.asarray(actions, dtype=jnp.int32),
        movement_codes=jnp.asarray(movement, dtype=jnp.int32),
        feeding_codes=jnp.asarray(feeding, dtype=jnp.int32),
        movement_energy_cost=jnp.float32(2),
        feeding_energy_cost=jnp.float32(3),
    )


def test_only_successful_outcomes_pay() -> None:
    result = apply(
        [
            ActionCode.MOVE_NORTH,
            ActionCode.STAY,
            ActionCode.EAT,
            ActionCode.EAT,
            ActionCode.REPRODUCE,
        ],
        [MovementResolutionCode.MOVED, MovementResolutionCode.CONFLICT_LOST, 0, 0, 0],
        [0, 0, FeedingResolutionCode.FED_FULL, FeedingResolutionCode.FED_PARTIAL, 0],
    )
    assert result.movement_cost_applied.tolist() == [2, 0, 0, 0, 0]
    assert result.feeding_cost_applied.tolist() == [0, 0, 3, 3, 0]
    assert result.population.energy.tolist() == [8, 10, 7, 7, 10]


def test_inconsistent_inputs_pay_nothing_and_are_counted() -> None:
    state = population([10, 10, 10, jnp.nan], [False, True, True, True])
    result = apply(
        [ActionCode.MOVE_NORTH, ActionCode.STAY, ActionCode.MOVE_EAST, ActionCode.STAY],
        [MovementResolutionCode.MOVED] * 4,
        [0, 0, FeedingResolutionCode.FED_FULL, 0],
        state,
    )
    assert result.invalid_action_cost_input_count.item() == 4
    assert result.action_cost_applied.tolist() == [0, 0, 0, 0]
    assert jnp.isnan(result.population.energy[3])


def test_phase_mismatches_and_unknown_codes_are_inconsistent() -> None:
    result = apply(
        [ActionCode.EAT, ActionCode.STAY, ActionCode.REPRODUCE],
        [MovementResolutionCode.CONFLICT_LOST, 99, MovementResolutionCode.NOT_MOVEMENT],
        [FeedingResolutionCode.FED_FULL, FeedingResolutionCode.NOT_FEEDING, 99],
    )
    assert result.invalid_action_cost_input_count.item() == 3
    assert result.action_cost_applied.tolist() == [0, 0, 0]
    assert result.population.energy.tolist() == [10, 10, 10]


def test_costs_are_id_independent_permutation_equivariant_and_jittable() -> None:
    state = population([8, 9], [True, True])
    first = apply([ActionCode.EAT, ActionCode.MOVE_WEST], [0, 1], [1, 0], state)
    changed_ids = eqx.tree_at(lambda p: p.agent_id, state, jnp.asarray([99, 42], dtype=jnp.int32))
    second = apply([ActionCode.EAT, ActionCode.MOVE_WEST], [0, 1], [1, 0], changed_ids)
    assert jnp.array_equal(first.action_cost_applied, second.action_cost_applied)
    compiled = eqx.filter_jit(apply_action_energy_costs)(
        population=state,
        actions_after_feeding=jnp.asarray([ActionCode.EAT, ActionCode.MOVE_WEST], jnp.int32),
        movement_codes=jnp.asarray([0, 1], jnp.int32),
        feeding_codes=jnp.asarray([1, 0], jnp.int32),
        movement_energy_cost=jnp.float32(2),
        feeding_energy_cost=jnp.float32(3),
    )
    assert jnp.array_equal(first.action_cost_applied, compiled.action_cost_applied)
    assert first.action_cost_applied[::-1].tolist() == [2, 3]
