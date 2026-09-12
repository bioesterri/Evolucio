"""Pure, identity-keyed mutation of fixed-capacity newborn genomes."""

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.dtypes import COUNT_DTYPE, REAL_DTYPE
from evolucio.core.policy.batch import GenomeBatch
from evolucio.core.policy.model import PolicyLinear
from evolucio.core.rng import derive_entity_keys
from evolucio.core.state import PopulationState
from evolucio.core.types import Array


class GenomeMutationStreamCode(IntEnum):
    """Stable independent substreams for every mask and noise array."""

    LAYER1_WEIGHT_MASK = 0
    LAYER1_WEIGHT_NOISE = 1
    LAYER1_BIAS_MASK = 2
    LAYER1_BIAS_NOISE = 3
    LAYER2_WEIGHT_MASK = 4
    LAYER2_WEIGHT_NOISE = 5
    LAYER2_BIAS_MASK = 6
    LAYER2_BIAS_NOISE = 7


class GenomeMutationResult(eqx.Module):
    """Mutated genomes and fixed-shape mutation audit summary."""

    genomes: GenomeBatch
    selected_parameter_count: Array
    effective_parameter_count: Array
    selected_weight_count: Array
    selected_bias_count: Array
    sum_abs_delta: Array
    max_abs_delta: Array
    mutated_newborn_count: Array
    invalid_newborn_genome_count: Array


def _substream_keys(entity_keys: Array, code: GenomeMutationStreamCode) -> Array:
    return jax.vmap(jax.random.fold_in, in_axes=(0, None))(entity_keys, int(code))


def _random_for_slots(keys: Array, shape: tuple[int, ...], *, normal: bool) -> Array:
    def sample_normal(key: Array) -> Array:
        return jax.random.normal(key, shape, dtype=REAL_DTYPE)  # pyright: ignore[reportUnknownMemberType]

    def sample_uniform(key: Array) -> Array:
        return jax.random.uniform(key, shape, dtype=REAL_DTYPE)  # pyright: ignore[reportUnknownMemberType]

    if normal:
        return jax.vmap(sample_normal)(keys)
    return jax.vmap(sample_uniform)(keys)


def _mutate_leaf(
    inherited: Array,
    entity_keys: Array,
    valid_newborn: Array,
    rate: Array,
    sigma: Array,
    abs_limit: Array,
    mask_code: GenomeMutationStreamCode,
    noise_code: GenomeMutationStreamCode,
) -> tuple[Array, Array, Array]:
    element_shape = inherited.shape[1:]
    selected = (
        _random_for_slots(_substream_keys(entity_keys, mask_code), element_shape, normal=False)
        < rate
    )
    selected &= valid_newborn.reshape((-1,) + (1,) * len(element_shape))
    noise = (
        _random_for_slots(_substream_keys(entity_keys, noise_code), element_shape, normal=True)
        * sigma
    )
    candidate = jnp.clip(inherited + selected * noise, -abs_limit, abs_limit)
    apply_mask = valid_newborn.reshape((-1,) + (1,) * len(element_shape))
    mutated = jnp.where(apply_mask, candidate, inherited)
    return mutated, selected, jnp.abs(mutated - inherited)


def mutate_newborn_genomes(
    *,
    genomes: GenomeBatch,
    population: PopulationState,
    newborn_mask: Array,
    genome_mutation_key: Array,
    weight_mutation_rate: Array,
    weight_mutation_sigma: Array,
    weight_abs_limit: Array,
    bias_mutation_rate: Array,
    bias_mutation_sigma: Array,
    bias_abs_limit: Array,
) -> GenomeMutationResult:
    """Mutate only valid newborn genomes using genome-identity-derived RNG."""
    finite = (
        jnp.all(jnp.isfinite(genomes.layer1.weight), axis=(1, 2))
        & jnp.all(jnp.isfinite(genomes.layer1.bias), axis=1)
        & jnp.all(jnp.isfinite(genomes.layer2.weight), axis=(1, 2))
        & jnp.all(jnp.isfinite(genomes.layer2.bias), axis=1)
    )
    weights_bounded = jnp.all(
        jnp.abs(genomes.layer1.weight) <= weight_abs_limit, axis=(1, 2)
    ) & jnp.all(jnp.abs(genomes.layer2.weight) <= weight_abs_limit, axis=(1, 2))
    biases_bounded = jnp.all(jnp.abs(genomes.layer1.bias) <= bias_abs_limit, axis=1) & jnp.all(
        jnp.abs(genomes.layer2.bias) <= bias_abs_limit, axis=1
    )
    consistent = (
        population.alive & (population.genome_id >= 0) & finite & weights_bounded & biases_bounded
    )
    valid_newborn = newborn_mask & consistent
    invalid_newborn = newborn_mask & ~consistent
    entity_keys = derive_entity_keys(genome_mutation_key, population.genome_id)

    l1w, l1w_selected, l1w_delta = _mutate_leaf(
        genomes.layer1.weight,
        entity_keys,
        valid_newborn,
        weight_mutation_rate,
        weight_mutation_sigma,
        weight_abs_limit,
        GenomeMutationStreamCode.LAYER1_WEIGHT_MASK,
        GenomeMutationStreamCode.LAYER1_WEIGHT_NOISE,
    )
    l1b, l1b_selected, l1b_delta = _mutate_leaf(
        genomes.layer1.bias,
        entity_keys,
        valid_newborn,
        bias_mutation_rate,
        bias_mutation_sigma,
        bias_abs_limit,
        GenomeMutationStreamCode.LAYER1_BIAS_MASK,
        GenomeMutationStreamCode.LAYER1_BIAS_NOISE,
    )
    l2w, l2w_selected, l2w_delta = _mutate_leaf(
        genomes.layer2.weight,
        entity_keys,
        valid_newborn,
        weight_mutation_rate,
        weight_mutation_sigma,
        weight_abs_limit,
        GenomeMutationStreamCode.LAYER2_WEIGHT_MASK,
        GenomeMutationStreamCode.LAYER2_WEIGHT_NOISE,
    )
    l2b, l2b_selected, l2b_delta = _mutate_leaf(
        genomes.layer2.bias,
        entity_keys,
        valid_newborn,
        bias_mutation_rate,
        bias_mutation_sigma,
        bias_abs_limit,
        GenomeMutationStreamCode.LAYER2_BIAS_MASK,
        GenomeMutationStreamCode.LAYER2_BIAS_NOISE,
    )

    def counts(first: Array, second: Array) -> Array:
        axes_first = tuple(range(1, first.ndim))
        axes_second = tuple(range(1, second.ndim))
        return jnp.sum(first, axis=axes_first, dtype=COUNT_DTYPE) + jnp.sum(
            second, axis=axes_second, dtype=COUNT_DTYPE
        )

    selected_weight_count = counts(l1w_selected, l2w_selected)
    selected_bias_count = counts(l1b_selected, l2b_selected)
    deltas = [
        l1w_delta.reshape((l1w_delta.shape[0], -1)),
        l1b_delta.reshape((l1b_delta.shape[0], -1)),
        l2w_delta.reshape((l2w_delta.shape[0], -1)),
        l2b_delta.reshape((l2b_delta.shape[0], -1)),
    ]
    all_delta = jnp.concatenate(deltas, axis=1)
    effective = all_delta != 0
    effective_parameter_count = jnp.sum(effective, axis=1, dtype=COUNT_DTYPE)
    return GenomeMutationResult(
        genomes=GenomeBatch(
            layer1=PolicyLinear(weight=l1w, bias=l1b),
            layer2=PolicyLinear(weight=l2w, bias=l2b),
        ),
        selected_parameter_count=selected_weight_count + selected_bias_count,
        effective_parameter_count=effective_parameter_count,
        selected_weight_count=selected_weight_count,
        selected_bias_count=selected_bias_count,
        sum_abs_delta=jnp.sum(all_delta, axis=1, dtype=REAL_DTYPE),
        max_abs_delta=jnp.max(all_delta, axis=1),
        mutated_newborn_count=jnp.sum(
            valid_newborn & (effective_parameter_count > 0), dtype=COUNT_DTYPE
        ),
        invalid_newborn_genome_count=jnp.sum(invalid_newborn, dtype=COUNT_DTYPE),
    )
