"""Basic deterministic compatibility tests for JAX and Equinox."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def test_jax_cpu_operation_and_jit() -> None:
    """JAX exposes a CPU and executes eager and compiled operations."""
    assert any(device.platform == "cpu" for device in jax.devices())

    values = jnp.array([1, 2, 3], dtype=jnp.int32)
    assert jnp.array_equal(values + 1, jnp.array([2, 3, 4], dtype=jnp.int32))

    double = jax.jit(lambda value: value * 2)
    assert int(double(jnp.array(4, dtype=jnp.int32))) == 8


def test_equinox_elementary_transformation() -> None:
    """Equinox can transform an elementary JAX-compatible operation."""
    increment = eqx.filter_jit(lambda value: value + 1)

    result = increment(jnp.array([1.0, 2.0], dtype=jnp.float32))

    assert jnp.array_equal(result, jnp.array([2.0, 3.0], dtype=jnp.float32))
