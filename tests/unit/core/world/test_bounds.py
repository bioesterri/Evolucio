import equinox as eqx
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

from evolucio.core import INDEX_DTYPE, MASK_DTYPE
from evolucio.core.world import positions_in_bounds


def test_closed_boundaries_are_vectorised_and_not_wrapped() -> None:
    positions = jnp.asarray(
        [
            [0, 0],
            [5, 0],
            [0, 3],
            [5, 3],
            [2, 1],
            [-1, 0],
            [0, -1],
            [6, 0],
            [0, 4],
            [6, 4],
            [-1, -1],
        ],
        dtype=INDEX_DTYPE,
    )
    before = positions.copy()
    expected = [True] * 5 + [False] * 6
    assert positions_in_bounds(positions, width=6, height=4).tolist() == expected
    assert jnp.array_equal(positions, before)


def test_shapes_dtype_and_eager_jit_equivalence() -> None:
    scalar = positions_in_bounds(jnp.asarray([2, 1]), width=6, height=4)
    batch = jnp.asarray([[[0, 0], [6, 0]], [[5, 3], [0, 4]]])
    eager = positions_in_bounds(batch, width=6, height=4)
    compiled = eqx.filter_jit(positions_in_bounds)(batch, width=6, height=4)
    assert scalar.shape == ()
    assert eager.shape == compiled.shape == (2, 2)
    assert eager.dtype == jnp.dtype(MASK_DTYPE)
    assert jnp.array_equal(eager, compiled)


@given(
    x=st.integers(-2, 8), y=st.integers(-2, 6), width=st.integers(1, 6), height=st.integers(1, 6)
)
@settings(max_examples=30)
def test_bounds_property(x: int, y: int, width: int, height: int) -> None:
    result = positions_in_bounds(jnp.asarray([x, y]), width=width, height=height)
    assert bool(result) is (0 <= x < width and 0 <= y < height)
