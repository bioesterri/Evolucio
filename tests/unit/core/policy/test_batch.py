import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from evolucio.core import REAL_DTYPE
from evolucio.core.policy import (
    GENOME_PARAMETER_COUNT,
    GenomeBatch,
    create_empty_genome_batch,
    policy_at,
    validate_genome_batch_structure,
    validate_policy_structure,
)


def test_empty_batch_has_exact_fixed_capacity_pytree() -> None:
    genomes = create_empty_genome_batch(5)
    leaves = jax.tree.leaves(genomes)
    assert isinstance(genomes, GenomeBatch)
    assert [leaf.shape for leaf in leaves] == [(5, 16, 15), (5, 16), (5, 7, 16), (5, 7)]
    assert len(leaves) == 4
    assert all(
        isinstance(leaf, jax.Array) and leaf.dtype == jnp.dtype(REAL_DTYPE) for leaf in leaves
    )
    assert all(bool(jnp.all(leaf == 0)) for leaf in leaves)
    assert sum(leaf.size for leaf in leaves) == 5 * GENOME_PARAMETER_COUNT
    validate_genome_batch_structure(genomes, max_agents=5)
    mapped = jax.tree.map(lambda leaf: leaf + 1, genomes)
    assert jax.tree.structure(mapped) == jax.tree.structure(genomes)
    assert [leaf.shape for leaf in jax.tree.leaves(mapped)] == [leaf.shape for leaf in leaves]


def test_empty_batch_and_policy_at_are_jittable() -> None:
    eager = create_empty_genome_batch(3)
    compiled = eqx.filter_jit(create_empty_genome_batch)(3)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, eager, compiled))
    policy = eqx.filter_jit(policy_at)(compiled, jnp.asarray(1, dtype=jnp.int32))
    validate_policy_structure(policy)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, policy, policy_at(eager, 1)))


def test_empty_batch_rejects_nonpositive_capacity() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        create_empty_genome_batch(0)
