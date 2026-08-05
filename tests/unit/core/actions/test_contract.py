import equinox as eqx
import jax.numpy as jnp

from evolucio.core.actions import (
    ACTION_DELTAS,
    ACTION_VALIDATION_CODE_COUNT,
    ActionValidationCode,
    action_deltas,
    is_movement_action,
    is_valid_action_code,
)
from evolucio.core.codes import ACTION_COUNT, ActionCode


def test_action_and_validation_codes_are_frozen() -> None:
    assert [(code.name, code.value) for code in ActionCode] == [
        ("STAY", 0),
        ("MOVE_NORTH", 1),
        ("MOVE_SOUTH", 2),
        ("MOVE_EAST", 3),
        ("MOVE_WEST", 4),
        ("EAT", 5),
        ("REPRODUCE", 6),
    ]
    assert ACTION_COUNT == 7
    assert [(code.name, code.value) for code in ActionValidationCode] == [
        ("ACCEPTED", 0),
        ("INACTIVE_SLOT", 1),
        ("INVALID_ACTION_CODE", 2),
        ("INVALID_ACTOR_POSITION", 3),
        ("MOVE_OUT_OF_BOUNDS", 4),
        ("EAT_NO_RESOURCE", 5),
    ]
    assert ACTION_VALIDATION_CODE_COUNT == len(ActionValidationCode) == 6
    assert len(ActionValidationCode.__members__) == 6


def test_action_deltas_and_classification_are_vectorized_and_jittable() -> None:
    assert ACTION_DELTAS == ((0, 0), (0, -1), (0, 1), (1, 0), (-1, 0), (0, 0), (0, 0))
    codes = jnp.asarray([-1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=jnp.int32)
    expected = jnp.asarray(
        [[0, 0], [0, 0], [0, -1], [0, 1], [1, 0], [-1, 0], [0, 0], [0, 0], [0, 0]],
        dtype=jnp.int32,
    )
    eager = action_deltas(codes)
    assert eager.shape == (9, 2)
    assert eager.dtype == jnp.int32
    assert jnp.array_equal(eager, expected)
    assert jnp.array_equal(eqx.filter_jit(action_deltas)(codes), eager)
    assert is_valid_action_code(codes).tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert is_movement_action(codes).tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert is_valid_action_code(codes).dtype == jnp.bool_
