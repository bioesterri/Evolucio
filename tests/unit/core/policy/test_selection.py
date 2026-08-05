import jax
import jax.numpy as jnp
import pytest

from evolucio.core.codes import ActionCode
from evolucio.core.policy import select_actions_deterministically


def test_exact_ties_negative_scores_and_action_order() -> None:
    scores = jnp.asarray(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [0, 5, 5, 0, 0, 0, 0],
            [0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 0, 5, 5],
            [-5, -3, -4, -1, -2, -6, -7],
        ],
        dtype=jnp.float32,
    )
    result = select_actions_deterministically(scores, jnp.ones(5, dtype=jnp.bool_))
    assert result.proposed_actions.tolist() == [0, 1, 3, 5, 3]
    assert int(result.exact_tie_count) == 4
    assert int(result.invalid_active_score_count) == 0
    assert result.proposed_actions.dtype == jnp.int32
    assert result.exact_tie_count.shape == () and result.exact_tie_count.dtype == jnp.int32


def test_every_action_code_can_be_proposed() -> None:
    scores = jnp.eye(7, dtype=jnp.float32)
    result = select_actions_deterministically(scores, jnp.ones(7, dtype=jnp.bool_))
    assert result.proposed_actions.tolist() == [int(code) for code in ActionCode]


def test_non_finite_active_rows_and_residual_inactive_rows_are_canonical() -> None:
    scores = jnp.asarray(
        [
            [0, 1, jnp.nan, 0, 0, 0, 0],
            [jnp.inf, 0, 0, 0, 0, 0, 0],
            [0, 0, -jnp.inf, 0, 0, 0, 0],
            [jnp.nan, jnp.inf, -jnp.inf, jnp.nan, jnp.inf, -jnp.inf, jnp.nan],
            [0, 0, 0, 0, 99, 0, jnp.nan],
            [0, 0, 0, 0, 0, 2, 0],
        ],
        dtype=jnp.float32,
    )
    alive = jnp.asarray([True, True, True, True, False, True])
    result = jax.jit(select_actions_deterministically)(scores, alive)
    assert int(result.invalid_active_score_count) == 4
    assert int(result.exact_tie_count) == 0
    assert jnp.array_equal(result.scores[:5], jnp.zeros((5, 7), dtype=jnp.float32))
    assert result.proposed_actions.tolist() == [0, 0, 0, 0, 0, 5]


@pytest.mark.parametrize("alive", [[], [False, False], [True, False], [True, True]])
def test_fixed_capacity_independent_of_live_count(alive: list[bool]) -> None:
    capacity = len(alive)
    if capacity == 0:
        return
    result = select_actions_deterministically(
        jnp.zeros((capacity, 7), dtype=jnp.float32), jnp.asarray(alive)
    )
    assert result.scores.shape == (capacity, 7)
    assert result.proposed_actions.shape == (capacity,)
    assert int(result.exact_tie_count) == sum(alive)


def test_selection_rejects_wrong_contract_dtypes() -> None:
    with pytest.raises(TypeError, match="raw_scores"):
        select_actions_deterministically(jnp.zeros((1, 7), dtype=jnp.int32), jnp.ones(1, bool))
    with pytest.raises(TypeError, match="alive"):
        select_actions_deterministically(jnp.zeros((1, 7), dtype=jnp.float32), jnp.ones(1, int))
