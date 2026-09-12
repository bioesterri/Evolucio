import jax
import jax.numpy as jnp

from evolucio.core.evolution import GenomeMutationResult, mutate_newborn_genomes
from evolucio.core.policy import GenomeBatch, create_empty_genome_batch
from evolucio.core.state import PopulationState
from evolucio.core.types import Array


def population(capacity: int = 4) -> PopulationState:
    ids = jnp.arange(capacity, dtype=jnp.int32)
    return PopulationState(
        alive=jnp.ones(capacity, dtype=jnp.bool_),
        agent_id=ids + 10,
        parent_id=jnp.full(capacity, -1, dtype=jnp.int32),
        lineage_id=ids,
        genome_id=ids + 100,
        generation=jnp.zeros(capacity, dtype=jnp.int32),
        position=jnp.zeros((capacity, 2), dtype=jnp.int32),
        energy=jnp.ones(capacity, dtype=jnp.float32),
        birth_step=jnp.zeros(capacity, dtype=jnp.int32),
        age=jnp.zeros(capacity, dtype=jnp.int32),
    )


def mutate(
    genomes: GenomeBatch,
    state: PopulationState,
    newborn_mask: Array,
    *,
    weight_rate: float = 1.0,
    bias_rate: float = 1.0,
    sigma: float = 0.1,
) -> GenomeMutationResult:
    return mutate_newborn_genomes(
        genomes=genomes,
        population=state,
        newborn_mask=newborn_mask,
        genome_mutation_key=jax.random.key(7),
        weight_mutation_rate=jnp.asarray(weight_rate, dtype=jnp.float32),
        weight_mutation_sigma=jnp.asarray(sigma, dtype=jnp.float32),
        weight_abs_limit=jnp.asarray(1.0, dtype=jnp.float32),
        bias_mutation_rate=jnp.asarray(bias_rate, dtype=jnp.float32),
        bias_mutation_sigma=jnp.asarray(sigma, dtype=jnp.float32),
        bias_abs_limit=jnp.asarray(0.5, dtype=jnp.float32),
    )


def leaves_equal(left: object, right: object) -> bool:
    return all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def test_empty_mask_and_zero_rates_are_exact_controls() -> None:
    genomes = create_empty_genome_batch(4)
    state = population()
    for mask, rates in [
        (jnp.zeros(4, dtype=jnp.bool_), (1.0, 1.0)),
        (jnp.array([False, True, False, True]), (0.0, 0.0)),
    ]:
        result = mutate(genomes, state, mask, weight_rate=rates[0], bias_rate=rates[1])
        assert leaves_equal(result.genomes, genomes)
        assert not bool(jnp.any(result.selected_parameter_count))
        assert not bool(jnp.any(result.effective_parameter_count))
        assert not bool(jnp.any(result.sum_abs_delta))
        assert not bool(jnp.any(result.max_abs_delta))


def test_rate_one_selects_each_enabled_group_and_preserves_non_newborns() -> None:
    genomes = create_empty_genome_batch(4)
    state = population()
    newborn = jnp.array([False, True, False, True])
    weights = mutate(genomes, state, newborn, bias_rate=0.0)
    biases = mutate(genomes, state, newborn, weight_rate=0.0)

    assert weights.selected_weight_count.tolist() == [0, 352, 0, 352]
    assert weights.selected_bias_count.tolist() == [0, 0, 0, 0]
    assert biases.selected_weight_count.tolist() == [0, 0, 0, 0]
    assert biases.selected_bias_count.tolist() == [0, 23, 0, 23]
    for result in (weights, biases):
        assert all(
            bool(jnp.array_equal(after[jnp.array([0, 2])], before[jnp.array([0, 2])]))
            for after, before in zip(
                jax.tree.leaves(result.genomes), jax.tree.leaves(genomes), strict=True
            )
        )
        assert bool(jnp.all(result.selected_parameter_count >= result.effective_parameter_count))
        assert bool(jnp.all(result.sum_abs_delta >= result.max_abs_delta))


def test_sigma_zero_selects_without_an_effective_change() -> None:
    result = mutate(
        create_empty_genome_batch(4), population(), jnp.ones(4, dtype=jnp.bool_), sigma=0.0
    )
    assert result.selected_parameter_count.tolist() == [375] * 4
    assert not bool(jnp.any(result.effective_parameter_count))
    assert int(result.mutated_newborn_count) == 0


def test_mutation_is_additive_clipped_and_summary_is_coherent() -> None:
    genomes = create_empty_genome_batch(4)
    genomes = jax.tree.map(lambda leaf: leaf + jnp.float32(0.45), genomes)
    result = mutate(genomes, population(), jnp.array([False, True, False, False]), sigma=10.0)
    assert bool(jnp.all(jnp.abs(result.genomes.layer1.weight[1]) <= 1.0))
    assert bool(jnp.all(jnp.abs(result.genomes.layer2.weight[1]) <= 1.0))
    assert bool(jnp.all(jnp.abs(result.genomes.layer1.bias[1]) <= 0.5))
    assert bool(jnp.all(jnp.abs(result.genomes.layer2.bias[1]) <= 0.5))
    assert int(result.selected_parameter_count[1]) == 375
    assert int(result.effective_parameter_count[1]) > 0
    assert float(result.sum_abs_delta[1]) >= float(result.max_abs_delta[1]) > 0
    assert int(result.mutated_newborn_count) == 1


def test_invalid_newborn_is_diagnosed_and_preserved_without_repair() -> None:
    genomes = create_empty_genome_batch(4)
    # Equinox modules are immutable containers, so replace the affected tree leaf.
    genomes = type(genomes)(
        layer1=type(genomes.layer1)(
            weight=genomes.layer1.weight.at[1, 0, 0].set(2.0), bias=genomes.layer1.bias
        ),
        layer2=genomes.layer2,
    )
    result = mutate(genomes, population(), jnp.array([False, True, False, False]))
    assert leaves_equal(result.genomes, genomes)
    assert int(result.invalid_newborn_genome_count) == 1
    assert int(result.selected_parameter_count[1]) == 0


def test_eager_jit_and_scan_have_fixed_equivalent_results() -> None:
    genomes = create_empty_genome_batch(4)
    state = population()
    newborn = jnp.array([False, True, False, True])
    eager = mutate(genomes, state, newborn)
    compiled = jax.jit(lambda batch: mutate(batch, state, newborn))(genomes)
    assert all(
        bool(jnp.allclose(a, b))
        for a, b in zip(jax.tree.leaves(eager), jax.tree.leaves(compiled), strict=True)
    )

    def body(batch: GenomeBatch, unused: None) -> tuple[GenomeBatch, Array]:
        result = mutate(batch, state, newborn, weight_rate=0.0, bias_rate=0.0)
        return result.genomes, result.selected_parameter_count

    final, counts = jax.lax.scan(body, genomes, xs=None, length=2)
    assert leaves_equal(final, genomes)
    assert counts.shape == (2, 4)
