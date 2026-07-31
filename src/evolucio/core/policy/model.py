"""Explicit, stateless Equinox policy model."""

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.dtypes import REAL_DTYPE
from evolucio.core.types import Array

from .schema import POLICY_PARAMETER_SPECS


class PolicyLinear(eqx.Module):
    """Minimal affine layer with explicit parameter leaves."""

    weight: Array
    bias: Array

    def __call__(self, inputs: Array) -> Array:
        return self.weight @ inputs + self.bias


class PolicyMLP(eqx.Module):
    """Fixed 15-16-7 policy producing linear action scores."""

    layer1: PolicyLinear
    layer2: PolicyLinear

    def __call__(self, observation: Array) -> Array:
        hidden = jax.nn.tanh(self.layer1(observation))
        return self.layer2(hidden)


def _validate_parameter(name: str, value: object, expected_shape: tuple[int, ...]) -> None:
    if not isinstance(value, jax.Array):
        raise TypeError(f"{name}: expected jax.Array, received {type(value).__name__}")
    if value.shape != expected_shape:
        raise ValueError(f"{name}: expected shape {expected_shape}, received {value.shape}")
    if value.dtype != jnp.dtype(REAL_DTYPE):
        raise TypeError(f"{name}: expected dtype float32, received {value.dtype}")
    if not bool(jnp.all(jnp.isfinite(value))):
        raise ValueError(f"{name}: contains NaN or infinite values")


def policy_from_parameters(
    *, layer1_weight: Array, layer1_bias: Array, layer2_weight: Array, layer2_bias: Array
) -> PolicyMLP:
    """Validate explicit host parameters and construct a policy without RNG."""
    values = (layer1_weight, layer1_bias, layer2_weight, layer2_bias)
    for spec, value in zip(POLICY_PARAMETER_SPECS, values, strict=True):
        _validate_parameter(spec.path, value, spec.shape)
    return PolicyMLP(
        layer1=PolicyLinear(weight=layer1_weight, bias=layer1_bias),
        layer2=PolicyLinear(weight=layer2_weight, bias=layer2_bias),
    )


def validate_policy_structure(policy: PolicyMLP) -> None:
    """Reject any policy that does not match the exact v1 structure."""
    if type(policy) is not PolicyMLP:
        raise TypeError(f"policy: expected PolicyMLP, received {type(policy).__name__}")
    if type(policy.layer1) is not PolicyLinear or type(policy.layer2) is not PolicyLinear:
        raise TypeError("policy: layer1 and layer2 must be PolicyLinear")
    leaves = jax.tree.leaves(policy)
    if len(leaves) != len(POLICY_PARAMETER_SPECS):
        raise ValueError(f"policy: expected four parameter leaves, received {len(leaves)}")
    for spec, value in zip(POLICY_PARAMETER_SPECS, leaves, strict=True):
        _validate_parameter(spec.path, value, spec.shape)
