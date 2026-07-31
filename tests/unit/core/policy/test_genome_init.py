import math

import equinox as eqx
import jax
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

from evolucio.core import COUNT_DTYPE, ID_DTYPE, MASK_DTYPE, NULL_ID, create_rng_state
from evolucio.core.policy import GenomeInitializationResult, initialize_genome_batch


def _initialize(alive: list[bool], ids: list[int], seed: int = 7) -> GenomeInitializationResult:
    return initialize_genome_batch(
        alive=jnp.asarray(alive, dtype=MASK_DTYPE),
        genome_ids=jnp.asarray(ids, dtype=ID_DTYPE),
        root_key=create_rng_state(seed).key,
    )


def test_glorot_bias_inactive_and_invalid_contract() -> None:
    result = _initialize([True, True, False, True], [10, 11, 99, NULL_ID])
    genomes = result.genomes
    assert result.invalid_active_genome_id_count.dtype == jnp.dtype(COUNT_DTYPE)
    assert int(result.invalid_active_genome_id_count) == 1
    assert bool(jnp.all(jnp.abs(genomes.layer1.weight[:2]) <= math.sqrt(6 / 31)))
    assert bool(jnp.all(jnp.abs(genomes.layer2.weight[:2]) <= math.sqrt(6 / 23)))
    assert not bool(jnp.all(genomes.layer1.weight[:2] == 0))
    assert bool(jnp.all(genomes.layer1.bias == 0))
    assert bool(jnp.all(genomes.layer2.bias == 0))
    for slot in (2, 3):
        assert all(bool(jnp.all(leaf[slot] == 0)) for leaf in jax.tree.leaves(genomes))


def test_determinism_identity_permutation_capacity_and_keys() -> None:
    first = _initialize([True, True, True, False], [20, 30, 40, 999])
    repeat = _initialize([True, True, True, False], [20, 30, 40, -1])
    assert jax.tree.all(jax.tree.map(jnp.array_equal, first.genomes, repeat.genomes))
    permutation = jnp.asarray([2, 0, 3, 1])
    permuted = _initialize([True, True, False, True], [40, 20, 999, 30])
    assert jax.tree.all(
        jax.tree.map(
            lambda left, right: jnp.array_equal(left[permutation], right),
            first.genomes,
            permuted.genomes,
        )
    )
    larger = _initialize([True, True, True, False, False], [20, 30, 40, -1, 500])
    assert jax.tree.all(
        jax.tree.map(lambda a, b: jnp.array_equal(a[:3], b[:3]), first.genomes, larger.genomes)
    )
    other_key = _initialize([True, True, True, False], [20, 30, 40, -1], seed=8)
    assert not jnp.array_equal(first.genomes.layer1.weight[:3], other_key.genomes.layer1.weight[:3])


def test_eager_and_jit_match_with_invalid_id() -> None:
    alive = jnp.asarray([True, False, True], dtype=MASK_DTYPE)
    ids = jnp.asarray([3, 77, -1], dtype=ID_DTYPE)
    key = create_rng_state(1).key
    eager = initialize_genome_batch(alive=alive, genome_ids=ids, root_key=key)
    compiled = eqx.filter_jit(initialize_genome_batch)(alive=alive, genome_ids=ids, root_key=key)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, eager, compiled))


@given(
    alive=st.lists(st.booleans(), min_size=1, max_size=6),
    seed=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=10, deadline=None)
def test_properties(alive: list[bool], seed: int) -> None:
    ids = list(range(len(alive)))
    if alive:
        ids[-1] = -1
    result = _initialize(alive, ids, seed)
    leaves = jax.tree.leaves(result.genomes)
    assert all(leaf.shape[0] == len(alive) and bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert int(result.invalid_active_genome_id_count) == int(alive[-1])
    for index, active in enumerate(alive):
        if not active or ids[index] < 0:
            assert all(bool(jnp.all(leaf[index] == 0)) for leaf in leaves)
