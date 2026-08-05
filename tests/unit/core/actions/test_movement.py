import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.actions import (
    ActionValidationCode,
    ActionValidationResult,
    MovementResolutionCode,
    MovementResolutionResult,
    resolve_cardinal_movement,
    validate_and_route_actions,
)
from evolucio.core.codes import ActionCode
from evolucio.core.dtypes import ID_DTYPE, INDEX_DTYPE, MASK_DTYPE
from evolucio.core.population import create_empty_population
from evolucio.core.rng import create_rng_state
from evolucio.core.spatial import rebuild_world_occupancy
from evolucio.core.state import PopulationState, WorldState


def _case(
    positions: list[list[int]], actions: list[int], *, ids: list[int] | None = None
) -> tuple[PopulationState, WorldState, ActionValidationResult]:
    capacity = len(positions)
    population = create_empty_population(capacity)
    population = eqx.tree_at(
        lambda state: (state.position, state.alive, state.agent_id),
        population,
        (
            jnp.asarray(positions, dtype=INDEX_DTYPE),
            jnp.ones((capacity,), dtype=MASK_DTYPE),
            jnp.asarray(ids if ids is not None else list(range(capacity)), dtype=ID_DTYPE),
        ),
    )
    world = WorldState(
        resources=jnp.ones((5, 5), dtype=jnp.float32),
        environment=jnp.zeros((5, 5), dtype=jnp.float32),
        occupancy=jnp.zeros((5, 5), dtype=jnp.int32),
    )
    world = rebuild_world_occupancy(world, population, width=5, height=5).world
    validation = validate_and_route_actions(
        proposed_actions=jnp.asarray(actions, dtype=jnp.int32),
        alive=population.alive,
        positions=population.position,
        resources=world.resources,
        width=5,
        height=5,
    )
    return population, world, validation


def _resolve(
    population: PopulationState, world: WorldState, validation: ActionValidationResult
) -> MovementResolutionResult:
    return resolve_cardinal_movement(
        population=population,
        world=world,
        action_validation=validation,
        movement_conflict_key=create_rng_state(7).key,
        width=5,
        height=5,
    )


def test_single_moves_non_movement_stays_and_state_is_preserved() -> None:
    population, world, validation = _case(
        [[2, 2], [0, 0], [4, 4]],
        [ActionCode.MOVE_NORTH, ActionCode.EAT, ActionCode.REPRODUCE],
    )
    result = _resolve(population, world, validation)
    assert result.population.position.tolist() == [[2, 1], [0, 0], [4, 4]]
    assert result.movement_codes.tolist() == [
        MovementResolutionCode.MOVED,
        MovementResolutionCode.NOT_MOVEMENT,
        MovementResolutionCode.NOT_MOVEMENT,
    ]
    assert result.actions_after_movement.tolist() == [
        ActionCode.MOVE_NORTH,
        ActionCode.EAT,
        ActionCode.REPRODUCE,
    ]
    assert jnp.array_equal(result.population.energy, population.energy)
    assert jnp.array_equal(result.world.resources, world.resources)
    assert int(result.world.occupancy.sum()) == 3


def test_snapshot_blocks_swap_and_a_cell_that_is_being_vacated() -> None:
    population, world, validation = _case(
        [[1, 2], [2, 2], [3, 2]],
        [ActionCode.MOVE_EAST, ActionCode.MOVE_EAST, ActionCode.MOVE_WEST],
    )
    result = _resolve(population, world, validation)
    assert result.population.position.tolist() == [[1, 2], [2, 2], [3, 2]]
    assert result.movement_codes.tolist() == [MovementResolutionCode.DESTINATION_OCCUPIED] * 3
    assert result.actions_after_movement.tolist() == [ActionCode.STAY] * 3


def test_conflict_has_one_identity_bound_winner_and_is_slot_equivariant() -> None:
    population, world, validation = _case(
        [[1, 2], [3, 2], [2, 3]],
        [ActionCode.MOVE_EAST, ActionCode.MOVE_WEST, ActionCode.MOVE_NORTH],
        ids=[91, 17, 44],
    )
    original = _resolve(population, world, validation)
    assert original.movement_codes.tolist().count(MovementResolutionCode.MOVED) == 1
    assert original.movement_codes.tolist().count(MovementResolutionCode.CONFLICT_LOST) == 2
    assert int(original.contested_destination_count) == 1
    permutation = jnp.asarray([2, 0, 1])
    permuted_population = jax.tree.map(lambda leaf: leaf[permutation], population)
    permuted_validation = jax.tree.map(lambda leaf: leaf[permutation], validation)
    permuted = _resolve(permuted_population, world, permuted_validation)
    inverse = jnp.argsort(permutation)
    assert jnp.array_equal(permuted.population.position[inverse], original.population.position)
    assert jnp.array_equal(permuted.movement_codes[inverse], original.movement_codes)
    assert jnp.array_equal(permuted.world.occupancy, original.world.occupancy)


def test_defensive_invalid_input_is_safe_and_falls_back_to_stay() -> None:
    population, world, validation = _case([[2, 2]], [ActionCode.MOVE_EAST])
    invalid = eqx.tree_at(
        lambda result: (result.validation_codes, result.move_targets),
        validation,
        (
            jnp.asarray([ActionValidationCode.EAT_NO_RESOURCE], dtype=jnp.int32),
            jnp.asarray([[99, 99]], dtype=jnp.int32),
        ),
    )
    result = _resolve(population, world, invalid)
    assert result.movement_codes.tolist() == [MovementResolutionCode.INVALID_MOVEMENT_INPUT]
    assert result.actions_after_movement.tolist() == [ActionCode.STAY]
    assert int(result.invalid_movement_input_count) == 1
    assert jnp.array_equal(result.population.position, population.position)
