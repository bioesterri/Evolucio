from itertools import pairwise

import equinox as eqx
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

from evolucio.core import REAL_DTYPE
from evolucio.core.world import regenerate_resources


def scalar(value: float) -> jnp.ndarray:
    return jnp.asarray(value, dtype=REAL_DTYPE)


def test_exact_regeneration_cases_and_jit() -> None:
    resources = jnp.asarray([[0.0, 0.5, 1.0]], dtype=REAL_DTYPE)
    kwargs = {
        "resource_capacity": scalar(1),
        "regeneration_rate": scalar(0.2),
        "regeneration_multiplier": scalar(0.5),
    }
    eager = regenerate_resources(resources, **kwargs)
    compiled = eqx.filter_jit(regenerate_resources)(resources, **kwargs)
    assert jnp.allclose(eager, jnp.asarray([[0.1, 0.55, 1.0]], dtype=REAL_DTYPE))
    assert jnp.array_equal(eager, compiled)
    assert eager.shape == resources.shape and eager.dtype == resources.dtype
    for field in ("regeneration_rate", "regeneration_multiplier"):
        zero = dict(kwargs)
        zero[field] = scalar(0)
        assert jnp.array_equal(regenerate_resources(resources, **zero), resources)


@given(
    values=st.lists(st.floats(0, 1, allow_nan=False), min_size=1, max_size=12),
    rate=st.floats(0, 0.99, allow_nan=False),
    multiplier=st.floats(0, 1, allow_nan=False),
)
@settings(max_examples=20, deadline=None)
def test_regeneration_properties(values: list[float], rate: float, multiplier: float) -> None:
    resources = jnp.asarray(values, dtype=REAL_DTYPE)
    result = regenerate_resources(
        resources,
        resource_capacity=scalar(1),
        regeneration_rate=scalar(rate),
        regeneration_multiplier=scalar(multiplier),
    )
    assert result.shape == resources.shape and result.dtype == resources.dtype
    assert bool(jnp.all(jnp.isfinite(result)))
    assert bool(jnp.all(result >= resources)) and bool(jnp.all(result <= 1))


def test_temporal_recovery_is_gradual() -> None:
    value = jnp.zeros((1, 1), dtype=REAL_DTYPE)
    observed = []
    for _ in range(5):
        value = regenerate_resources(
            value,
            resource_capacity=scalar(1),
            regeneration_rate=scalar(0.2),
            regeneration_multiplier=scalar(1),
        )
        observed.append(float(value[0, 0]))
    assert all(left < right for left, right in pairwise(observed))
    assert observed[-1] < 1
