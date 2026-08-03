import ast
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from evolucio.core import (
    ID_DTYPE,
    PRNG_IMPLEMENTATION,
    RNG_STREAM_COUNT,
    RngState,
    RngStreamCode,
    advance_rng,
    create_rng_state,
    derive_entity_keys,
    derive_indexed_key,
    derive_stream_key,
)


def key_data(key: jax.Array) -> jax.Array:
    return jax.random.key_data(key)


def test_create_typed_root_key_is_explicit_and_deterministic() -> None:
    first = create_rng_state(0)
    same = create_rng_state(0)
    other = create_rng_state(1)

    assert isinstance(first, RngState)
    assert isinstance(first.key, jax.Array)
    assert first.key.shape == ()
    assert jax.dtypes.issubdtype(first.key.dtype, jax.dtypes.prng_key)
    assert jax.random.key_impl(first.key) == PRNG_IMPLEMENTATION
    assert jnp.array_equal(key_data(first.key), key_data(same.key))
    assert not jnp.array_equal(key_data(first.key), key_data(other.key))
    assert len(jax.tree.leaves(first)) == 1


@pytest.mark.parametrize("seed", [True, 1.0, "1", None])
def test_create_rng_rejects_non_python_integer_seed(seed: object) -> None:
    with pytest.raises(TypeError):
        create_rng_state(seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_create_rng_respects_host_seed_range(seed: int) -> None:
    with pytest.raises(ValueError):
        create_rng_state(seed)


def test_advance_order_is_pure_reproducible_and_jittable() -> None:
    original = create_rng_state(42)
    original_data = key_data(original.key)
    eager_next, eager_step = advance_rng(original)
    compiled_next, compiled_step = eqx.filter_jit(advance_rng)(original)
    expected_next, expected_step = jax.random.split(original.key, num=2)

    assert jnp.array_equal(key_data(original.key), original_data)
    assert jnp.array_equal(key_data(eager_next.key), key_data(expected_next))
    assert jnp.array_equal(key_data(eager_step), key_data(expected_step))
    assert jnp.array_equal(key_data(eager_next.key), key_data(compiled_next.key))
    assert jnp.array_equal(key_data(eager_step), key_data(compiled_step))
    assert not jnp.array_equal(key_data(eager_next.key), key_data(eager_step))
    later, later_step = advance_rng(eager_next)
    assert not jnp.array_equal(key_data(later.key), key_data(eager_next.key))
    assert not jnp.array_equal(key_data(later_step), key_data(eager_step))
    assert len(jax.tree.leaves(eager_next)) == 1


def test_stream_codes_are_exact_stable_and_without_aliases() -> None:
    expected = {
        "WORLD_INITIALIZATION": 0,
        "RESOURCE_INITIALIZATION": 1,
        "AGENT_INITIALIZATION": 2,
        "GENOME_INITIALIZATION": 3,
        "ENVIRONMENT_UPDATE": 4,
        "ACTION_TIE_BREAK": 5,
        "MOVEMENT_CONFLICT": 6,
        "RESOURCE_CONFLICT": 7,
        "REPRODUCTION_CONFLICT": 8,
        "BIRTH_PLACEMENT": 9,
        "GENOME_MUTATION": 10,
    }
    assert {item.name: int(item) for item in RngStreamCode} == expected
    assert [int(item) for item in RngStreamCode] == list(range(11))
    assert len(RngStreamCode.__members__) == RNG_STREAM_COUNT == 11


def test_stream_derivation_is_repeatable_separate_and_order_independent() -> None:
    _, step_key = advance_rng(create_rng_state(7))
    environment_a = derive_stream_key(step_key, RngStreamCode.ENVIRONMENT_UPDATE)
    mutation_a = derive_stream_key(step_key, RngStreamCode.GENOME_MUTATION)
    mutation_b = derive_stream_key(step_key, RngStreamCode.GENOME_MUTATION)
    environment_b = derive_stream_key(step_key, RngStreamCode.ENVIRONMENT_UPDATE)

    assert jnp.array_equal(key_data(environment_a), key_data(environment_b))
    assert jnp.array_equal(key_data(mutation_a), key_data(mutation_b))
    assert not jnp.array_equal(key_data(environment_a), key_data(mutation_a))
    assert jax.random.uniform(environment_a) == jax.random.uniform(environment_b)
    assert jax.random.uniform(mutation_a) == jax.random.uniform(mutation_b)
    with pytest.raises(TypeError):
        derive_stream_key(step_key, "ENVIRONMENT_UPDATE")  # type: ignore[arg-type]


def test_indexed_and_entity_keys_follow_identity_not_position() -> None:
    _, step_key = advance_rng(create_rng_state(8))
    base = derive_stream_key(step_key, RngStreamCode.GENOME_MUTATION)
    ids = jnp.asarray([10, 20, 30], dtype=ID_DTYPE)
    original = derive_entity_keys(base, ids)
    repeated = derive_entity_keys(base, ids)
    reordered = derive_entity_keys(base, jnp.asarray([30, 10, 20], dtype=ID_DTYPE))
    extended = derive_entity_keys(base, jnp.asarray([10, 20, 30, 40], dtype=ID_DTYPE))
    compiled = jax.jit(derive_entity_keys)(base, ids)

    assert original.shape == (3,)
    assert jax.dtypes.issubdtype(original.dtype, jax.dtypes.prng_key)
    assert jnp.array_equal(key_data(original), key_data(repeated))
    assert jnp.array_equal(key_data(reordered), key_data(original)[jnp.asarray([2, 0, 1])])
    assert jnp.array_equal(key_data(extended[:3]), key_data(original))
    assert jnp.array_equal(key_data(compiled), key_data(original))
    assert len({tuple(row) for row in key_data(original).tolist()}) == 3
    assert jnp.array_equal(key_data(derive_indexed_key(base, 10)), key_data(original[0]))


def test_index_validation() -> None:
    key = create_rng_state(0).key
    with pytest.raises(ValueError):
        derive_indexed_key(key, jnp.asarray([1], dtype=ID_DTYPE))
    with pytest.raises(TypeError):
        derive_indexed_key(key, jnp.asarray(1, dtype=jnp.int16))
    with pytest.raises(ValueError):
        derive_entity_keys(key, jnp.asarray([[1]], dtype=ID_DTYPE))
    with pytest.raises(TypeError):
        derive_entity_keys(key, jnp.asarray([1], dtype=jnp.int16))


def test_core_uses_no_forbidden_random_sources() -> None:
    forbidden = ("jax.random.PRNGKey", "numpy.random", "np.random")
    for path in Path("src/evolucio/core").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden)
        tree = ast.parse(source)
        imports = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        imports += [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        assert "random" not in imports
