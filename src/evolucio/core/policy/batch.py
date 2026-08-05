"""Fixed-capacity batched neural genome representation."""

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.dtypes import REAL_DTYPE
from evolucio.core.types import Array

from .model import PolicyLinear, PolicyMLP
from .schema import POLICY_PARAMETER_COUNT, POLICY_PARAMETER_SPECS


class GenomeBatch(eqx.Module):
    """Four policy parameter leaves stacked on the population axis."""

    layer1: PolicyLinear
    layer2: PolicyLinear


def create_empty_genome_batch(max_agents: int) -> GenomeBatch:
    """Create the canonical all-zero genome batch."""
    if max_agents <= 0:
        raise ValueError("max_agents must be greater than zero")
    return GenomeBatch(
        layer1=PolicyLinear(
            weight=jnp.zeros((max_agents, 16, 15), dtype=REAL_DTYPE),  # pyright: ignore[reportUnknownMemberType]
            bias=jnp.zeros((max_agents, 16), dtype=REAL_DTYPE),  # pyright: ignore[reportUnknownMemberType]
        ),
        layer2=PolicyLinear(
            weight=jnp.zeros((max_agents, 7, 16), dtype=REAL_DTYPE),  # pyright: ignore[reportUnknownMemberType]
            bias=jnp.zeros((max_agents, 7), dtype=REAL_DTYPE),  # pyright: ignore[reportUnknownMemberType]
        ),
    )


def policy_at(genomes: GenomeBatch, slot_index: Array | int) -> PolicyMLP:
    """Expose one individual policy for inspection and testing."""
    return PolicyMLP(
        layer1=PolicyLinear(
            weight=genomes.layer1.weight[slot_index], bias=genomes.layer1.bias[slot_index]
        ),
        layer2=PolicyLinear(
            weight=genomes.layer2.weight[slot_index], bias=genomes.layer2.bias[slot_index]
        ),
    )


def validate_genome_batch_structure(genomes: GenomeBatch, *, max_agents: int) -> None:
    """Validate the exact host-side structure of a genome batch."""
    if type(genomes) is not GenomeBatch:
        raise TypeError(f"genomes: expected GenomeBatch, received {type(genomes).__name__}")
    if type(genomes.layer1) is not PolicyLinear or type(genomes.layer2) is not PolicyLinear:
        raise TypeError("genomes: layer1 and layer2 must be PolicyLinear")
    leaves = jax.tree.leaves(genomes)
    if len(leaves) != 4:
        raise ValueError(f"genomes: expected four parameter leaves, received {len(leaves)}")
    total = 0
    for spec, value in zip(POLICY_PARAMETER_SPECS, leaves, strict=True):
        expected = (max_agents, *spec.shape)
        if not isinstance(value, jax.Array):
            raise TypeError(f"{spec.path}: expected jax.Array")
        if value.shape != expected:
            raise ValueError(f"{spec.path}: expected shape {expected}, received {value.shape}")
        if value.dtype != jnp.dtype(REAL_DTYPE):
            raise TypeError(f"{spec.path}: expected dtype float32, received {value.dtype}")
        if not bool(jnp.all(jnp.isfinite(value))):
            raise ValueError(f"{spec.path}: contains NaN or infinite values")
        total += value.size
    if total != max_agents * POLICY_PARAMETER_COUNT:
        raise ValueError("genomes: parameter count does not match the policy schema")
