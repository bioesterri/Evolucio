import equinox as eqx
import jax.numpy as jnp

from evolucio.core import INDEX_DTYPE, REAL_DTYPE
from evolucio.core.observations.gather import build_cardinal_ray_positions, gather_map_values


def test_gather_uses_xy_without_clipping_or_wrapping() -> None:
    values = jnp.arange(12, dtype=REAL_DTYPE).reshape(3, 4)
    positions = jnp.asarray([[2, 1], [-1, 1], [4, 1], [0, 3]], dtype=INDEX_DTYPE)
    expected = jnp.asarray([6.0, -9.0, -9.0, -9.0], dtype=REAL_DTYPE)
    assert jnp.array_equal(
        gather_map_values(values, positions, width=4, height=3, fill_value=-9), expected
    )
    assert jnp.array_equal(
        eqx.filter_jit(gather_map_values)(values, positions, width=4, height=3, fill_value=-9),
        expected,
    )


def test_cardinal_ray_order_and_shape() -> None:
    positions = jnp.asarray([[3, 4], [0, 0]], dtype=INDEX_DTYPE)
    rays = build_cardinal_ray_positions(positions, radius=2)
    assert rays.shape == (2, 4, 2, 2)
    assert jnp.array_equal(
        rays[0],
        jnp.asarray([[[3, 3], [3, 2]], [[3, 5], [3, 6]], [[4, 4], [5, 4]], [[2, 4], [1, 4]]]),
    )
