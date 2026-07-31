import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from evolucio.config import ExperimentConfig, WorldCoreConfig, compile_config, load_config
from evolucio.core import (
    COUNT_DTYPE,
    REAL_DTYPE,
    RngStreamCode,
    WorldState,
    create_rng_state,
    derive_stream_key,
)
from evolucio.core.world import initialize_environment, initialize_resources, initialize_world


@pytest.fixture
def config() -> ExperimentConfig:
    return load_config("tests/fixtures/config/valid_v1.yaml")


def changed(config: ExperimentConfig, **values: object) -> ExperimentConfig:
    raw = config.model_dump(mode="python")
    raw["world"].update(values)  # type: ignore[union-attr]
    return ExperimentConfig.model_validate(raw)


def world_config(config: ExperimentConfig, **values: object) -> WorldCoreConfig:
    return compile_config(changed(config, **values)).core.world


def test_uniform_resources_contract_and_seed_independence(config: ExperimentConfig) -> None:
    core = world_config(config, width=6, height=4, resource_distribution="uniform")
    first = initialize_resources(core, create_rng_state(1).key)
    second = initialize_resources(core, create_rng_state(2).key)
    compiled = eqx.filter_jit(initialize_resources)(core, create_rng_state(1).key)
    assert first.shape == (4, 6) and first.dtype == jnp.dtype(REAL_DTYPE)
    assert jnp.all(first == core.initial_resource_mean)
    assert float(jnp.var(first)) == 0.0
    assert jnp.array_equal(first, second)
    assert jnp.array_equal(first, compiled)


@pytest.mark.parametrize("mean", [0.0, 10.0])
def test_uniform_resource_extremes(config: ExperimentConfig, mean: float) -> None:
    core = world_config(config, resource_distribution="uniform", initial_resource_mean=mean)
    assert jnp.all(initialize_resources(core, create_rng_state(0).key) == mean)


def test_patch_resources_are_bounded_centered_variable_and_reproducible(
    config: ExperimentConfig,
) -> None:
    core = world_config(
        config, width=9, height=7, initial_resource_mean=4.0, resource_patch_count=4
    )
    key = create_rng_state(10).key
    first = initialize_resources(core, key)
    same = initialize_resources(core, key)
    other = initialize_resources(core, create_rng_state(11).key)
    compiled = eqx.filter_jit(initialize_resources)(core, key)
    assert first.shape == (7, 9) and first.dtype == jnp.dtype(REAL_DTYPE)
    assert bool(jnp.all(jnp.isfinite(first)))
    assert float(first.min()) >= 0 and float(first.max()) <= float(core.resource_capacity)
    assert float(first.mean()) == pytest.approx(4.0, abs=2e-5)
    assert float(first.var()) > 0
    assert jnp.allclose(first, same) and jnp.allclose(first, compiled)
    assert not jnp.allclose(first, other)


def test_patch_parameters_control_structure_without_changing_shape(
    config: ExperimentConfig,
) -> None:
    key = create_rng_state(4).key
    zero = initialize_resources(world_config(config, resource_patch_contrast=0.0), key)
    low = initialize_resources(world_config(config, resource_patch_contrast=0.2), key)
    high = initialize_resources(world_config(config, resource_patch_contrast=0.9), key)
    wide = initialize_resources(world_config(config, resource_patch_radius=9.0), key)
    assert jnp.all(zero == 5.0)
    assert float(jnp.var(high)) >= float(jnp.var(low))
    assert wide.shape == high.shape and not jnp.allclose(wide, high)


def test_environment_and_complete_world_contract(config: ExperimentConfig) -> None:
    core = world_config(config, width=6, height=4, environment_initial_value=0.3)
    root = create_rng_state(12).key
    root_before = jax.random.key_data(root).copy()
    environment = initialize_environment(core)
    eager = initialize_world(core, root)
    compiled = eqx.filter_jit(initialize_world)(core, root)
    assert isinstance(eager, WorldState)
    assert environment.shape == (4, 6) and environment.dtype == jnp.dtype(REAL_DTYPE)
    assert jnp.allclose(environment, 0.3)
    assert eager.resources.dtype == eager.environment.dtype == jnp.dtype(REAL_DTYPE)
    assert eager.occupancy.dtype == jnp.dtype(COUNT_DTYPE) and jnp.all(eager.occupancy == 0)
    assert jax.tree.structure(eager) == jax.tree.structure(compiled)
    assert jax.tree.all(jax.tree.map(jnp.allclose, eager, compiled))
    assert jnp.array_equal(jax.random.key_data(root), root_before)


def test_world_stream_is_order_independent(config: ExperimentConfig) -> None:
    core = compile_config(config).core.world
    root = create_rng_state(22).key
    first = initialize_world(core, root)
    _ = derive_stream_key(root, RngStreamCode.AGENT_INITIALIZATION)
    second = initialize_world(core, root)
    assert jnp.array_equal(first.resources, second.resources)


def test_uniform_and_patches_have_comparable_totals_and_distinct_signatures(
    config: ExperimentConfig,
) -> None:
    uniform_host = changed(config, resource_distribution="uniform")
    patches_host = changed(config, resource_distribution="patches")
    uniform = compile_config(uniform_host)
    patches = compile_config(patches_host)
    key = create_rng_state(7).key
    a = initialize_world(uniform.core.world, key).resources
    b = initialize_world(patches.core.world, key).resources
    assert a.shape == b.shape and a.dtype == b.dtype
    assert float(a.mean()) == pytest.approx(float(b.mean()), abs=2e-5)
    assert float(a.sum()) == pytest.approx(float(b.sum()), abs=5e-3)
    assert float(a.var()) == 0 and float(b.var()) > 0
    assert uniform.config_hash != patches.config_hash
    assert uniform.compile_signature != patches.compile_signature


def test_dynamic_world_fields_preserve_signature(config: ExperimentConfig) -> None:
    baseline = compile_config(config)
    for field, value in [
        ("resource_capacity", 12.0),
        ("initial_resource_mean", 4.0),
        ("resource_patch_radius", 3.0),
        ("resource_patch_contrast", 0.4),
        ("environment_initial_value", 0.4),
    ]:
        candidate = compile_config(changed(config, **{field: value}))
        assert candidate.config_hash != baseline.config_hash
        assert candidate.compile_signature == baseline.compile_signature
        assert jax.tree.structure(candidate.core) == jax.tree.structure(baseline.core)


@given(width=st.integers(1, 6), height=st.integers(1, 6), contrast=st.floats(0, 1, allow_nan=False))
@settings(
    max_examples=12, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_small_world_properties(
    config: ExperimentConfig, width: int, height: int, contrast: float
) -> None:
    core = world_config(
        config, width=width, height=height, resource_patch_count=2, resource_patch_contrast=contrast
    )
    world = initialize_world(core, create_rng_state(3).key)
    assert world.resources.shape == (height, width)
    assert bool(jnp.all(jnp.isfinite(world.resources)))
    assert bool(jnp.all((world.resources >= 0) & (world.resources <= core.resource_capacity)))
    assert float(world.resources.mean()) == pytest.approx(
        float(core.initial_resource_mean), abs=2e-5
    )
    assert bool(jnp.all((world.environment >= 0) & (world.environment <= 1)))
    assert bool(jnp.all(world.occupancy == 0))
