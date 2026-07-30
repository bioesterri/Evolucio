from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from evolucio.core.policy import (
    PolicyLinear,
    PolicyMLP,
    policy_from_parameters,
    validate_policy_structure,
)


def parameters() -> dict[str, jax.Array]:
    return {
        "layer1_weight": jnp.zeros((16, 15), dtype=jnp.float32),
        "layer1_bias": jnp.zeros((16,), dtype=jnp.float32),
        "layer2_weight": jnp.zeros((7, 16), dtype=jnp.float32),
        "layer2_bias": jnp.zeros((7,), dtype=jnp.float32),
    }


def test_construction_pytree_and_validation() -> None:
    policy = policy_from_parameters(**parameters())
    validate_policy_structure(policy)
    leaves, treedef = jax.tree.flatten(policy)
    assert isinstance(policy, PolicyMLP) and isinstance(policy.layer1, PolicyLinear)
    assert [x.shape for x in leaves] == [(16, 15), (16,), (7, 16), (7,)]
    assert all(isinstance(x, jax.Array) and x.dtype == jnp.float32 for x in leaves)
    assert jax.tree.structure(jax.tree.map(lambda x: x, policy)) == treedef
    assert jax.tree.structure(policy_from_parameters(**parameters())) == treedef
    arrays, remainder = eqx.partition(policy, eqx.is_array)
    assert len(jax.tree.leaves(arrays)) == 4 and not jax.tree.leaves(remainder)


@pytest.mark.parametrize(
    "name,shape",
    [
        ("layer1_weight", (15, 16)),
        ("layer1_bias", (15,)),
        ("layer2_weight", (16, 7)),
        ("layer2_bias", (8,)),
    ],
)
def test_rejects_wrong_shapes(name: str, shape: tuple[int, ...]) -> None:
    values = parameters()
    values[name] = jnp.zeros(shape, dtype=jnp.float32)
    with pytest.raises(ValueError, match=rf"{name.replace('_', '.')}.*expected shape.*received"):
        policy_from_parameters(**values)


def test_rejects_type_dtype_and_nonfinite() -> None:
    values = parameters()
    values["layer1_weight"] = 1  # type: ignore[assignment]
    with pytest.raises(TypeError, match=r"layer1\.weight"):
        policy_from_parameters(**values)
    values = parameters()
    values["layer1_bias"] = jnp.zeros((16,), dtype=jnp.int32)
    with pytest.raises(TypeError, match=r"expected dtype float32.*int32"):
        policy_from_parameters(**values)
    for bad in (jnp.nan, jnp.inf):
        values = parameters()
        values["layer2_bias"] = values["layer2_bias"].at[0].set(bad)
        with pytest.raises(ValueError, match=r"layer2\.bias.*NaN or infinite"):
            policy_from_parameters(**values)


def test_zero_policy_eager_jit_vmap_and_purity() -> None:
    policy = policy_from_parameters(**parameters())
    observation = jnp.arange(15, dtype=jnp.float32)
    before = [x.copy() for x in jax.tree.leaves(policy)]
    observation_before = observation.copy()
    eager = policy(observation)
    compiled = eqx.filter_jit(lambda p, x: p(x))(policy, observation)
    batch = jnp.stack((observation, -observation))
    mapped = jax.vmap(policy)(batch)
    assert eager.shape == compiled.shape == (7,) and eager.dtype == jnp.float32
    assert jnp.array_equal(eager, jnp.zeros(7)) and jnp.array_equal(eager, compiled)
    assert jnp.array_equal(mapped, jnp.stack([policy(x) for x in batch]))
    assert all(jnp.array_equal(x, y) for x, y in zip(before, jax.tree.leaves(policy), strict=True))
    assert jnp.array_equal(observation, observation_before)


def test_linear_output_manual_forward_and_sensitivity() -> None:
    values = parameters()
    values["layer1_weight"] = values["layer1_weight"].at[0, 0].set(2.0).at[1, 1].set(-1.0)
    values["layer1_bias"] = values["layer1_bias"].at[:3].set(jnp.array([0.5, -0.25, 0.0]))
    values["layer2_weight"] = values["layer2_weight"].at[0, 0].set(3.0).at[1, 1].set(-2.0)
    values["layer2_bias"] = jnp.array([-2.0, 0.0, 3.0, 1.5, -0.5, 4.0, 2.0], dtype=jnp.float32)
    policy = policy_from_parameters(**values)
    observation = jnp.arange(15, dtype=jnp.float32) / 10
    hidden_pre = values["layer1_weight"] @ observation + values["layer1_bias"]
    expected = values["layer2_weight"] @ jnp.tanh(hidden_pre) + values["layer2_bias"]
    assert jnp.allclose(policy(observation), expected)
    assert not jnp.array_equal(policy(observation), policy(observation.at[0].set(1.0)))
    assert policy(observation)[0] < 0 and not jnp.isclose(policy(observation).sum(), 1.0)
    assert jnp.array_equal(policy(observation), policy(observation))


def test_output_bias_is_not_activated() -> None:
    values = parameters()
    values["layer2_bias"] = jnp.array([-2, 0, 3, 1.5, -0.5, 4, 2], dtype=jnp.float32)
    policy = policy_from_parameters(**values)
    assert jnp.array_equal(policy(jnp.zeros(15, dtype=jnp.float32)), values["layer2_bias"])


def test_policy_source_excludes_prohibited_components() -> None:
    source = "\n".join(path.read_text() for path in Path("src/evolucio/core/policy").glob("*.py"))
    prohibited = (
        "eqx.nn.MLP",
        "eqx.nn.Sequential",
        "jax.random",
        "numpy.random",
        "random.",
        "optax",
        "optimizer",
        "dropout",
        "batchnorm",
        "recurrent",
        "lstm",
        "gru",
        "softmax",
        "argmax",
    )
    assert not [term for term in prohibited if term in source.lower()]
