import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

from evolucio.core.actions import (
    ActionValidationCode,
    ActionValidationResult,
    validate_and_route_actions,
)
from evolucio.core.codes import ActionCode
from evolucio.core.types import Array


def _validate(
    proposed: list[int], alive: list[bool], positions: list[list[int]], resources: list[list[float]]
) -> ActionValidationResult:
    resource_array = jnp.asarray(resources, dtype=jnp.float32)
    return validate_and_route_actions(
        proposed_actions=jnp.asarray(proposed, dtype=jnp.int32),
        alive=jnp.asarray(alive, dtype=jnp.bool_),
        positions=jnp.asarray(positions, dtype=jnp.int32),
        resources=resource_array,
        width=resource_array.shape[1],
        height=resource_array.shape[0],
    )


def test_all_seven_actions_are_accepted_at_an_interior_resource_cell() -> None:
    positions = [[2, 2]] * 7
    resources = [[1.0] * 5 for _ in range(5)]
    result = _validate(list(range(7)), [True] * 7, positions, resources)
    assert result.proposed_actions.tolist() == list(range(7))
    assert result.routed_actions.tolist() == list(range(7))
    assert result.validation_codes.tolist() == [ActionValidationCode.ACCEPTED] * 7
    assert result.move_targets.tolist() == [[2, 2], [2, 1], [2, 3], [3, 2], [1, 2], [2, 2], [2, 2]]


def test_precedence_inactive_invalid_position_bounds_resource_and_accepted() -> None:
    result = _validate(
        [99, 99, ActionCode.MOVE_NORTH, ActionCode.MOVE_NORTH, ActionCode.EAT, ActionCode.STAY],
        [False, True, True, True, True, True],
        [[-1, -1], [-1, -1], [-1, 0], [1, 0], [1, 1], [1, 1]],
        [[0.0] * 3 for _ in range(3)],
    )
    assert result.validation_codes.tolist() == [
        ActionValidationCode.INACTIVE_SLOT,
        ActionValidationCode.INVALID_ACTION_CODE,
        ActionValidationCode.INVALID_ACTOR_POSITION,
        ActionValidationCode.MOVE_OUT_OF_BOUNDS,
        ActionValidationCode.EAT_NO_RESOURCE,
        ActionValidationCode.ACCEPTED,
    ]
    assert result.routed_actions.tolist() == [ActionCode.STAY] * 6
    assert result.move_targets.tolist() == [[-1, -1], [-1, -1], [-1, 0], [1, 0], [1, 1], [1, 1]]


def test_invalid_codes_preserve_proposals_and_are_safe_under_jit() -> None:
    proposed = jnp.asarray([-10, -1, 7, 8, 100], dtype=jnp.int32)
    positions = jnp.asarray([[1, 1]] * 5, dtype=jnp.int32)
    resources = jnp.ones((3, 3), dtype=jnp.float32)
    function = eqx.filter_jit(validate_and_route_actions)
    result = function(
        proposed_actions=proposed,
        alive=jnp.ones((5,), dtype=jnp.bool_),
        positions=positions,
        resources=resources,
        width=3,
        height=3,
    )
    assert jnp.array_equal(result.proposed_actions, proposed)
    assert result.routed_actions.tolist() == [ActionCode.STAY] * 5
    assert result.validation_codes.tolist() == [ActionValidationCode.INVALID_ACTION_CODE] * 5
    assert jnp.array_equal(result.move_targets, positions)


def test_closed_rectangular_boundaries_do_not_wrap() -> None:
    result = _validate(
        [
            ActionCode.MOVE_NORTH,
            ActionCode.MOVE_SOUTH,
            ActionCode.MOVE_EAST,
            ActionCode.MOVE_WEST,
            ActionCode.MOVE_EAST,
        ],
        [True] * 5,
        [[1, 0], [1, 2], [3, 1], [0, 1], [0, 0]],
        [[1.0] * 4 for _ in range(3)],
    )
    assert result.validation_codes.tolist() == [
        ActionValidationCode.MOVE_OUT_OF_BOUNDS,
        ActionValidationCode.MOVE_OUT_OF_BOUNDS,
        ActionValidationCode.MOVE_OUT_OF_BOUNDS,
        ActionValidationCode.MOVE_OUT_OF_BOUNDS,
        ActionValidationCode.ACCEPTED,
    ]
    assert result.move_targets.tolist() == [[1, 0], [1, 2], [3, 1], [0, 1], [1, 0]]


def test_local_feeding_and_deferred_reproduction_have_no_effects() -> None:
    resources = jnp.asarray([[0.0, 0.25], [2.0, 0.0]], dtype=jnp.float32)
    before = resources.copy()
    result = validate_and_route_actions(
        proposed_actions=jnp.asarray([ActionCode.EAT, ActionCode.EAT, ActionCode.REPRODUCE]),
        alive=jnp.ones((3,), dtype=jnp.bool_),
        positions=jnp.asarray([[0, 0], [1, 0], [0, 1]], dtype=jnp.int32),
        resources=resources,
        width=2,
        height=2,
    )
    assert result.routed_actions.tolist() == [ActionCode.STAY, ActionCode.EAT, ActionCode.REPRODUCE]
    assert result.validation_codes.tolist() == [
        ActionValidationCode.EAT_NO_RESOURCE,
        ActionValidationCode.ACCEPTED,
        ActionValidationCode.ACCEPTED,
    ]
    assert jnp.array_equal(resources, before)


def test_multiple_local_intents_are_not_resolved() -> None:
    result = _validate(
        [ActionCode.MOVE_EAST, ActionCode.MOVE_WEST, ActionCode.EAT, ActionCode.EAT],
        [True] * 4,
        [[0, 1], [2, 1], [1, 1], [1, 1]],
        [[0.1] * 3 for _ in range(3)],
    )
    assert result.routed_actions.tolist() == [
        ActionCode.MOVE_EAST,
        ActionCode.MOVE_WEST,
        ActionCode.EAT,
        ActionCode.EAT,
    ]
    assert result.move_targets.tolist()[:2] == [[1, 1], [1, 1]]


def test_fixed_shapes_zero_some_and_all_alive() -> None:
    trees = []
    for alive in ([False] * 4, [True, False, True, False], [True] * 4):
        result = _validate([0, 1, 5, 6], alive, [[1, 1]] * 4, [[1.0] * 3 for _ in range(3)])
        assert [leaf.shape for leaf in jax.tree.leaves(result)] == [(4,), (4,), (4,), (4, 2)]
        trees.append(jax.tree.structure(result))
    assert trees[0] == trees[1] == trees[2]


def test_validation_is_equivariant_to_slot_permutation() -> None:
    proposed = jnp.asarray([1, 5, 6, 99], dtype=jnp.int32)
    alive = jnp.asarray([True, True, False, True])
    positions = jnp.asarray([[1, 1], [0, 0], [-1, -1], [2, 1]], dtype=jnp.int32)
    resources = jnp.asarray([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=jnp.float32)
    permutation = jnp.asarray([3, 0, 2, 1])
    original = validate_and_route_actions(
        proposed_actions=proposed,
        alive=alive,
        positions=positions,
        resources=resources,
        width=3,
        height=2,
    )
    permuted = validate_and_route_actions(
        proposed_actions=proposed[permutation],
        alive=alive[permutation],
        positions=positions[permutation],
        resources=resources,
        width=3,
        height=2,
    )
    for original_leaf, permuted_leaf in zip(
        jax.tree.leaves(original), jax.tree.leaves(permuted), strict=True
    ):
        assert jnp.array_equal(permuted_leaf, original_leaf[permutation])


def test_validation_runs_inside_lax_scan() -> None:
    proposals = jnp.asarray([[0, 1], [5, 6], [2, 4]], dtype=jnp.int32)
    alive = jnp.asarray([True, True])
    positions = jnp.asarray([[1, 1], [2, 1]], dtype=jnp.int32)
    resources = jnp.ones((3, 4), dtype=jnp.float32)

    def body(carry: None, proposed: Array) -> tuple[None, Array]:
        result = validate_and_route_actions(
            proposed_actions=proposed,
            alive=alive,
            positions=positions,
            resources=resources,
            width=4,
            height=3,
        )
        return carry, result.routed_actions

    _, scanned = jax.lax.scan(body, None, proposals)
    eager = jnp.stack([body(None, proposed)[1] for proposed in proposals])
    assert jnp.array_equal(scanned, eager)


def test_public_validation_api_is_narrow() -> None:
    assert list(inspect.signature(validate_and_route_actions).parameters) == [
        "proposed_actions",
        "alive",
        "positions",
        "resources",
        "width",
        "height",
    ]


@settings(max_examples=30, deadline=None)
@given(st.lists(st.integers(-3, 10), min_size=1, max_size=8))
def test_routing_properties_for_arbitrary_codes(proposed: list[int]) -> None:
    capacity = len(proposed)
    positions = [[index % 3, index % 2] for index in range(capacity)]
    alive = [(index % 3) != 0 for index in range(capacity)]
    result = _validate(proposed, alive, positions, [[0.0, 1.0, 0.5], [1.0, 0.0, 1.0]])
    assert all(0 <= action < 7 for action in result.routed_actions.tolist())
    for index, code in enumerate(result.validation_codes.tolist()):
        if code != ActionValidationCode.ACCEPTED:
            assert result.routed_actions[index] == ActionCode.STAY
        else:
            assert result.routed_actions[index] == proposed[index]
        if not alive[index]:
            assert code == ActionValidationCode.INACTIVE_SLOT


def test_new_modules_exclude_prohibited_patterns() -> None:
    import evolucio.core.actions.contract as contract
    import evolucio.core.actions.validate as validate

    source = inspect.getsource(contract) + inspect.getsource(validate)
    prohibited = (
        "jax.random",
        "numpy.random",
        "np.random",
        "random.",
        "second_best",
        "softmax",
        "scores",
        "for agent in",
        "for slot in",
    )
    assert not any(pattern in source for pattern in prohibited)
