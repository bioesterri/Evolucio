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
        agent_id=ids,
        parent_id=-jnp.ones(capacity, dtype=jnp.int32),
        lineage_id=ids,
        genome_id=ids + 100,
        generation=jnp.zeros(capacity, dtype=jnp.int32),
        position=jnp.zeros((capacity, 2), dtype=jnp.int32),
        energy=jnp.ones(capacity, dtype=jnp.float32),
        birth_step=jnp.zeros(capacity, dtype=jnp.int32),
        age=jnp.zeros(capacity, dtype=jnp.int32),
    )


def mutate(
    genomes: GenomeBatch, state: PopulationState, newborn_mask: Array
) -> GenomeMutationResult:
    return mutate_newborn_genomes(
        genomes=genomes,
        population=state,
        newborn_mask=newborn_mask,
        genome_mutation_key=jax.random.key(7),
        weight_mutation_rate=jnp.float32(1),
        weight_mutation_sigma=jnp.float32(0.1),
        weight_abs_limit=jnp.float32(1),
        bias_mutation_rate=jnp.float32(1),
        bias_mutation_sigma=jnp.float32(0.1),
        bias_abs_limit=jnp.float32(1),
    )


def leaves_equal(left: object, right: object) -> bool:
    return all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def test_identity_rng_is_reproducible_distinct_and_slot_equivariant() -> None:
    genomes = create_empty_genome_batch(4)
    state = population()
    newborn = jnp.ones(4, dtype=jnp.bool_)
    first = mutate(genomes, state, newborn)
    second = mutate(genomes, state, newborn)
    assert leaves_equal(first, second)
    assert not bool(jnp.array_equal(first.genomes.layer1.weight[0], first.genomes.layer1.weight[1]))

    permutation = jnp.array([2, 0, 3, 1])
    inverse = jnp.argsort(permutation)
    permuted_genomes = jax.tree.map(lambda leaf: leaf[permutation], genomes)
    permuted_state = jax.tree.map(lambda leaf: leaf[permutation], state)
    permuted = mutate(permuted_genomes, permuted_state, newborn[permutation])
    restored = jax.tree.map(lambda leaf: leaf[inverse], permuted.genomes)
    assert leaves_equal(restored, first.genomes)


def test_different_base_key_changes_mutation() -> None:
    genomes = create_empty_genome_batch(4)
    state = population()
    newborn = jnp.ones(4, dtype=jnp.bool_)
    first = mutate(genomes, state, newborn)
    second = mutate_newborn_genomes(
        genomes=genomes,
        population=state,
        newborn_mask=newborn,
        genome_mutation_key=jax.random.key(8),
        weight_mutation_rate=jnp.float32(1),
        weight_mutation_sigma=jnp.float32(0.1),
        weight_abs_limit=jnp.float32(1),
        bias_mutation_rate=jnp.float32(1),
        bias_mutation_sigma=jnp.float32(0.1),
        bias_abs_limit=jnp.float32(1),
    )
    assert not leaves_equal(first.genomes, second.genomes)
