"""Fixed discrete-action and local-validation vocabulary."""

# pyright: reportUnknownMemberType=false

from enum import IntEnum

import jax.numpy as jnp

from evolucio.core.codes import ACTION_COUNT, ActionCode
from evolucio.core.dtypes import CODE_DTYPE, INDEX_DTYPE, MASK_DTYPE
from evolucio.core.types import Array


class ActionValidationCode(IntEnum):
    """Stable reasons produced by preliminary local validation."""

    ACCEPTED = 0
    INACTIVE_SLOT = 1
    INVALID_ACTION_CODE = 2
    INVALID_ACTOR_POSITION = 3
    MOVE_OUT_OF_BOUNDS = 4
    EAT_NO_RESOURCE = 5


ACTION_VALIDATION_CODE_COUNT = len(ActionValidationCode)

ACTION_DELTAS = (
    (0, 0),
    (0, -1),
    (0, 1),
    (1, 0),
    (-1, 0),
    (0, 0),
    (0, 0),
)


def is_valid_action_code(action_codes: Array) -> Array:
    """Classify codes in the contiguous ``ActionCode`` range."""
    return jnp.asarray((action_codes >= 0) & (action_codes < ACTION_COUNT), dtype=MASK_DTYPE)


def is_movement_action(action_codes: Array) -> Array:
    """Classify the four cardinal movement codes."""
    return jnp.asarray(
        (action_codes >= ActionCode.MOVE_NORTH) & (action_codes <= ActionCode.MOVE_WEST),
        dtype=MASK_DTYPE,
    )


def action_deltas(action_codes: Array) -> Array:
    """Return safe ``[dx, dy]`` deltas, using zero for unknown codes."""
    valid_codes = is_valid_action_code(action_codes)
    stay = jnp.asarray(ActionCode.STAY, dtype=CODE_DTYPE)
    safe_actions = jnp.where(valid_codes, action_codes, stay)
    delta_table = jnp.asarray(ACTION_DELTAS, dtype=INDEX_DTYPE)
    return delta_table[safe_actions]
