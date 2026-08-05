import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from evolucio.core.dtypes import REAL_DTYPE
from evolucio.core.policy import (
    GenomeBatch,
    PolicyLinear,
    create_empty_genome_batch,
    infer_and_select_actions,
    infer_policy_scores,
    policy_at,
)


def _known_genomes(capacity: int) -> GenomeBatch:
    values1 = jnp.arange(capacity * 16 * 15, dtype=REAL_DTYPE).reshape(capacity, 16, 15)
    values2 = jnp.arange(capacity * 7 * 16, dtype=REAL_DTYPE).reshape(capacity, 7, 16)
    return GenomeBatch(
        layer1=PolicyLinear(weight=(values1 - 100.0) / 500.0, bias=jnp.zeros((capacity, 16))),
        layer2=PolicyLinear(weight=(values2 - 50.0) / 300.0, bias=jnp.zeros((capacity, 7))),
    )


def test_batched_inference_matches_each_policy_and_jit() -> None:
    genomes = _known_genomes(3)
    observations = jnp.arange(45, dtype=REAL_DTYPE).reshape(3, 15) / 50.0
    actual = infer_policy_scores(genomes, observations)
    reference = jnp.stack([policy_at(genomes, slot)(observations[slot]) for slot in range(3)])
    jitted = jax.jit(infer_policy_scores)(genomes, observations)
    assert actual.shape == (3, 7) and actual.dtype == jnp.float32
    assert jnp.allclose(actual, reference, rtol=2e-6, atol=2e-6)
    assert jnp.allclose(jitted, actual)


def test_manual_formula_and_linear_output_bias() -> None:
    genomes = create_empty_genome_batch(1)
    genomes = eqx.tree_at(
        lambda batch: (batch.layer1.weight, batch.layer2.weight, batch.layer2.bias),
        genomes,
        (
            jnp.ones((1, 16, 15), dtype=REAL_DTYPE),
            jnp.ones((1, 7, 16), dtype=REAL_DTYPE),
            jnp.arange(7, dtype=REAL_DTYPE)[None, :],
        ),
    )
    observations = jnp.full((1, 15), 0.1, dtype=REAL_DTYPE)
    expected = jnp.full((1, 7), 16 * jnp.tanh(jnp.float32(1.5))) + jnp.arange(7)[None, :]
    assert jnp.allclose(infer_policy_scores(genomes, observations), expected)


def test_integrated_api_scan_and_slot_permutation() -> None:
    genomes = _known_genomes(3)
    observations = jnp.arange(45, dtype=REAL_DTYPE).reshape(3, 15) / 50.0
    alive = jnp.asarray([True, False, True])
    eager = infer_and_select_actions(genomes, observations, alive)
    compiled = eqx.filter_jit(infer_and_select_actions)(genomes, observations, alive)
    assert jax.tree.structure(eager) == jax.tree.structure(compiled)
    assert jnp.allclose(eager.scores, compiled.scores)

    sequence = jnp.stack([observations, observations * 0.5])
    _, scanned = jax.lax.scan(
        lambda carry, item: (carry, infer_and_select_actions(genomes, item, alive)),
        None,
        sequence,
    )
    expected = jnp.stack(
        [infer_and_select_actions(genomes, item, alive).scores for item in sequence]
    )
    assert jnp.allclose(scanned.scores, expected)

    permutation = jnp.asarray([2, 0, 1])
    permuted_genomes = jax.tree.map(lambda leaf: leaf[permutation], genomes)
    permuted = infer_and_select_actions(
        permuted_genomes, observations[permutation], alive[permutation]
    )
    assert jnp.allclose(permuted.scores, eager.scores[permutation])
    assert jnp.array_equal(permuted.proposed_actions, eager.proposed_actions[permutation])
    assert permuted.exact_tie_count == eager.exact_tie_count


def test_inference_rejects_wrong_observation_dtype() -> None:
    with pytest.raises(TypeError, match="dtype float32"):
        infer_policy_scores(create_empty_genome_batch(1), jnp.zeros((1, 15), dtype=jnp.int32))


def test_production_inference_is_structurally_vectorized() -> None:
    source = inspect.getsource(inspect.getmodule(infer_policy_scores))
    for forbidden in ("for slot in", "for agent in", "policy_at(", "jnp.vectorize"):
        assert forbidden not in source
