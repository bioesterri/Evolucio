import jax.numpy as jnp

from evolucio.core.actions.movement import _resolve_claims


def test_injected_priority_collision_has_no_positional_fallback() -> None:
    eligible = jnp.asarray([True, True, False])
    targets = jnp.asarray([4, 4, 0], dtype=jnp.int32)
    priorities = tuple(jnp.asarray([value, value, 99], dtype=jnp.uint32) for value in (10, 20, 30))
    winner, collision, claims, finalists = _resolve_claims(
        eligible, targets, priorities, cell_count=9
    )
    assert winner.tolist() == [False, False, False]
    assert collision.tolist() == [True, True, False]
    assert int(claims[4]) == int(finalists[4]) == 2
