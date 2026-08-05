"""Deterministic founder-genome initialization."""

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.codes import RngStreamCode
from evolucio.core.dtypes import COUNT_DTYPE, REAL_DTYPE
from evolucio.core.rng import derive_entity_keys, derive_indexed_key, derive_stream_key
from evolucio.core.types import Array

from .batch import GenomeBatch
from .model import PolicyLinear, PolicyMLP
from .schema import POLICY_HIDDEN_SIZE, POLICY_INPUT_SIZE, POLICY_OUTPUT_SIZE


class GenomeInitializationStreamCode(IntEnum):
    """Stable parameter substreams within one genome."""

    LAYER1_WEIGHT = 0
    LAYER2_WEIGHT = 1


class GenomeInitializationResult(eqx.Module):
    """Initialized genomes and a JIT-safe invalid-identity diagnostic."""

    genomes: GenomeBatch
    invalid_active_genome_id_count: Array


def glorot_uniform(key: Array, *, out_features: int, in_features: int) -> Array:
    """Sample one float32 matrix with Glorot uniform initialization."""
    denominator = jnp.asarray(in_features + out_features, dtype=REAL_DTYPE)  # pyright: ignore[reportUnknownMemberType]
    limit = jnp.sqrt(jnp.asarray(6, dtype=REAL_DTYPE) / denominator)  # pyright: ignore[reportUnknownMemberType]
    return jax.random.uniform(  # pyright: ignore[reportUnknownMemberType]
        key,
        (out_features, in_features),
        dtype=REAL_DTYPE,
        minval=-limit,
        maxval=limit,
    )


def _initialize_active(key: Array) -> PolicyMLP:
    layer1_key = derive_indexed_key(key, GenomeInitializationStreamCode.LAYER1_WEIGHT)
    layer2_key = derive_indexed_key(key, GenomeInitializationStreamCode.LAYER2_WEIGHT)
    return PolicyMLP(
        layer1=PolicyLinear(
            weight=glorot_uniform(
                layer1_key, out_features=POLICY_HIDDEN_SIZE, in_features=POLICY_INPUT_SIZE
            ),
            bias=jnp.zeros((POLICY_HIDDEN_SIZE,), dtype=REAL_DTYPE),  # pyright: ignore[reportUnknownMemberType]
        ),
        layer2=PolicyLinear(
            weight=glorot_uniform(
                layer2_key, out_features=POLICY_OUTPUT_SIZE, in_features=POLICY_HIDDEN_SIZE
            ),
            bias=jnp.zeros((POLICY_OUTPUT_SIZE,), dtype=REAL_DTYPE),  # pyright: ignore[reportUnknownMemberType]
        ),
    )


def _zero_policy(key: Array) -> PolicyMLP:
    del key
    return PolicyMLP(
        layer1=PolicyLinear(
            weight=jnp.zeros((POLICY_HIDDEN_SIZE, POLICY_INPUT_SIZE), dtype=REAL_DTYPE),  # pyright: ignore[reportUnknownMemberType]
            bias=jnp.zeros((POLICY_HIDDEN_SIZE,), dtype=REAL_DTYPE),  # pyright: ignore[reportUnknownMemberType]
        ),
        layer2=PolicyLinear(
            weight=jnp.zeros((POLICY_OUTPUT_SIZE, POLICY_HIDDEN_SIZE), dtype=REAL_DTYPE),  # pyright: ignore[reportUnknownMemberType]
            bias=jnp.zeros((POLICY_OUTPUT_SIZE,), dtype=REAL_DTYPE),  # pyright: ignore[reportUnknownMemberType]
        ),
    )


def initialize_genome_batch(
    *, alive: Array, genome_ids: Array, root_key: Array
) -> GenomeInitializationResult:
    """Initialize fixed-capacity genomes from persistent genome identities."""
    valid_ids = genome_ids >= 0
    safe_ids = jnp.where(valid_ids, genome_ids, jnp.zeros_like(genome_ids))  # pyright: ignore[reportUnknownMemberType]
    stream_key = derive_stream_key(root_key, RngStreamCode.GENOME_INITIALIZATION)
    entity_keys = derive_entity_keys(stream_key, safe_ids)

    def initialize_one(key: Array, enabled: Array) -> PolicyMLP:
        return jax.lax.cond(enabled, _initialize_active, _zero_policy, key)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    policies = jax.vmap(initialize_one)(entity_keys, alive & valid_ids)
    genomes = GenomeBatch(layer1=policies.layer1, layer2=policies.layer2)
    invalid_count = jnp.sum(alive & ~valid_ids, dtype=COUNT_DTYPE)
    return GenomeInitializationResult(genomes=genomes, invalid_active_genome_id_count=invalid_count)
