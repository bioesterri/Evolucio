import jax.numpy as jnp

from evolucio.core import REAL_DTYPE
from evolucio.core.observations.normalize import normalize_positive, normalize_signed_margin


def test_positive_normalization_clips_and_handles_zero_scale() -> None:
    values = jnp.asarray([-1.0, 0.0, 5.0, 10.0, 20.0], dtype=REAL_DTYPE)
    result = normalize_positive(values, jnp.asarray(10.0, dtype=REAL_DTYPE))
    assert result.dtype == jnp.dtype(REAL_DTYPE)
    assert jnp.array_equal(result, jnp.asarray([0, 0, 0.5, 1, 1], dtype=REAL_DTYPE))
    zero = normalize_positive(values, jnp.asarray(0.0, dtype=REAL_DTYPE))
    assert jnp.array_equal(zero, jnp.zeros_like(values))
    assert not jnp.any(jnp.isnan(zero))


def test_signed_margin_clips_and_handles_zero_scale() -> None:
    values = jnp.asarray([-20.0, 30.0, 40.0, 50.0, 200.0], dtype=REAL_DTYPE)
    result = normalize_signed_margin(values, jnp.asarray(40.0), jnp.asarray(100.0))
    assert jnp.allclose(result, jnp.asarray([-0.6, -0.1, 0.0, 0.1, 1.0]))
    assert jnp.array_equal(
        normalize_signed_margin(values, jnp.asarray(40.0), jnp.asarray(0.0)), jnp.zeros_like(values)
    )
