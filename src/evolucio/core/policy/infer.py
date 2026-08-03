"""Population-vectorized policy inference."""

import jax
import jax.numpy as jnp

from evolucio.core.dtypes import REAL_DTYPE
from evolucio.core.types import Array

from .batch import GenomeBatch
from .schema import POLICY_INPUT_SIZE, POLICY_OUTPUT_SIZE
from .selection import PolicyDecisionResult, select_actions_deterministically


def infer_policy_scores(genomes: GenomeBatch, observations: Array) -> Array:
    """Evaluate every genome directly over the fixed population axis."""
    capacity = genomes.layer1.weight.shape[0]
    if observations.shape != (capacity, POLICY_INPUT_SIZE):
        raise ValueError(
            f"observations: expected shape ({capacity}, {POLICY_INPUT_SIZE}), "
            f"received {observations.shape}"
        )
    if observations.dtype != jnp.dtype(REAL_DTYPE):
        raise TypeError(f"observations: expected dtype float32, received {observations.dtype}")
    expected_shapes = (
        (capacity, 16, POLICY_INPUT_SIZE),
        (capacity, 16),
        (capacity, POLICY_OUTPUT_SIZE, 16),
        (capacity, POLICY_OUTPUT_SIZE),
    )
    leaves = (
        genomes.layer1.weight,
        genomes.layer1.bias,
        genomes.layer2.weight,
        genomes.layer2.bias,
    )
    for value, expected in zip(leaves, expected_shapes, strict=True):
        if value.shape != expected:
            raise ValueError(f"genome leaf: expected shape {expected}, received {value.shape}")
        if value.dtype != jnp.dtype(REAL_DTYPE):
            raise TypeError(f"genome leaf: expected dtype float32, received {value.dtype}")

    hidden = jax.nn.tanh(
        jnp.einsum(  # pyright: ignore[reportUnknownMemberType]
            "chi,ci->ch", genomes.layer1.weight, observations
        )
        + genomes.layer1.bias
    )
    return (
        jnp.einsum(  # pyright: ignore[reportUnknownMemberType]
            "coh,ch->co", genomes.layer2.weight, hidden
        )
        + genomes.layer2.bias
    )


def infer_and_select_actions(
    genomes: GenomeBatch, observations: Array, alive: Array
) -> PolicyDecisionResult:
    """Infer raw scores and deterministically produce action proposals."""
    return select_actions_deterministically(infer_policy_scores(genomes, observations), alive)
