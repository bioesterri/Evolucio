import dataclasses
import re
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from pydantic import BaseModel

from evolucio.config import (
    COMPILE_SIGNATURE_SCHEMA_VERSION,
    CompiledConfig,
    CompileSignature,
    ConfigCompilationError,
    CoreConfig,
    ExperimentConfig,
    build_compile_signature,
    compile_config,
    compile_signature_digest,
    freeze_config,
)


def replace(config: ExperimentConfig, block: str, **changes: object) -> ExperimentConfig:
    """Validate a copy with changes restricted to one configuration block."""
    data = config.model_dump(mode="python")
    value = data[block]
    assert isinstance(value, dict)
    value.update(changes)
    return ExperimentConfig.model_validate(data)


def test_basic_compilation_preserves_full_config_identity(config: ExperimentConfig) -> None:
    before = config.model_dump(mode="python")
    compiled = compile_config(config)

    assert isinstance(compiled, CompiledConfig)
    assert isinstance(compiled.core, CoreConfig)
    assert isinstance(compiled.compile_signature, CompileSignature)
    assert compiled.config_hash == freeze_config(config).config_hash
    assert config.model_dump(mode="python") == before


def test_dynamic_arrays_have_explicit_core_dtypes(config: ExperimentConfig) -> None:
    core = compile_config(config).core
    arrays = [leaf for leaf in jax.tree.leaves(core) if eqx.is_array(leaf)]

    assert arrays
    assert core.population.initial_agents.dtype == jnp.int32
    assert core.energy.basal_cost.dtype == jnp.float32
    assert all(array.dtype not in {jnp.dtype("float64"), jnp.dtype("int64")} for array in arrays)
    assert all(array.shape == () for array in arrays if array.size == 1)
    assert all(
        array.dtype in {jnp.dtype("float32"), jnp.dtype("int32"), jnp.dtype("bool")}
        for array in arrays
    )


def test_dynamic_change_preserves_signature_and_tree(config: ExperimentConfig) -> None:
    changed = replace(config, "energy", basal_cost=0.25, movement_cost=0.15)
    first = compile_config(config)
    second = compile_config(changed)

    assert first.config_hash != second.config_hash
    assert first.compile_signature == second.compile_signature
    assert compile_signature_digest(first.compile_signature) == compile_signature_digest(
        second.compile_signature
    )
    assert jax.tree.structure(first.core) == jax.tree.structure(second.core)


@pytest.mark.parametrize(
    ("block", "change"),
    [
        ("world", {"width": 65}),
        ("world", {"height": 65}),
        ("world", {"resource_distribution": "uniform"}),
        ("world", {"resource_patch_count": 9}),
        ("population", {"max_agents": 1025}),
        ("population", {"max_births_per_step": 65}),
        ("runtime", {"chunk_size": 64}),
    ],
)
def test_static_change_changes_signature(
    config: ExperimentConfig, block: str, change: dict[str, object]
) -> None:
    assert build_compile_signature(config) != build_compile_signature(
        replace(config, block, **change)
    )


def test_policy_static_fields_are_in_signature(config: ExperimentConfig) -> None:
    signature = build_compile_signature(config)
    assert signature.hidden_size == config.policy.hidden_size
    assert signature.observation_schema_version == config.policy.observation_schema_version
    assert signature.action_schema_version == config.policy.action_schema_version
    assert signature.activation == config.policy.activation


@pytest.mark.parametrize(
    ("block", "change"),
    [
        ("runtime", {"steps": 20_000}),
        ("persistence", {"output_dir": "other-runs"}),
        ("persistence", {"batch_size": 2048}),
    ],
)
def test_host_only_change_is_excluded(
    config: ExperimentConfig, block: str, change: dict[str, object]
) -> None:
    changed = replace(config, block, **change)
    first = compile_config(config)
    second = compile_config(changed)

    assert first.config_hash != second.config_hash
    assert first.compile_signature == second.compile_signature
    leaves = jax.tree.leaves(second.core)
    non_array_leaves = [leaf for leaf in leaves if not eqx.is_array(leaf)]
    assert change[next(iter(change))] not in non_array_leaves
    assert not any(isinstance(leaf, (Path, BaseModel, dict, list)) for leaf in leaves)


def test_seed_is_host_only(config: ExperimentConfig) -> None:
    changed = config.model_copy(update={"seed": 99})
    assert compile_config(config).compile_signature == compile_config(changed).compile_signature


def test_prng_implementation_versions_compile_signature(config: ExperimentConfig) -> None:
    signature = build_compile_signature(config)
    assert COMPILE_SIGNATURE_SCHEMA_VERSION == signature.signature_schema_version == 3
    assert signature.rng_implementation == "threefry2x32"
    assert "seed" not in {field.name for field in dataclasses.fields(signature)}


def test_signature_is_hashable_and_digest_is_stable(config: ExperimentConfig) -> None:
    signature = build_compile_signature(config)
    cache: dict[CompileSignature, object] = {}
    cache[signature] = object()
    reconstructed = CompileSignature(
        **{
            field.name: getattr(signature, field.name)
            for field in reversed(dataclasses.fields(signature))
        }
    )
    digest = compile_signature_digest(signature)

    assert cache[reconstructed] is cache[signature]
    assert digest == compile_signature_digest(reconstructed)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    changed = replace(config, "world", width=65)
    assert digest != compile_signature_digest(build_compile_signature(changed))


def test_environment_schedule_arrays_and_shape_signature(config: ExperimentConfig) -> None:
    phase = {
        "start_step": 0,
        "end_step": 10,
        "regeneration_multiplier": 0.75,
        "stress_level": 0.25,
    }
    changed = replace(config, "world", environment_schedule=(phase,))
    compiled = compile_config(changed)

    assert compiled.compile_signature.environment_schedule_length == 1
    calendar = compiled.core.world.environment_calendar
    assert calendar.phase_count == 1
    assert calendar.start_steps.shape == (1,)
    assert calendar.start_steps.dtype == jnp.int32
    assert calendar.environment_values.dtype == jnp.float32
    assert compiled.compile_signature != compile_config(config).compile_signature


def test_core_is_valid_pytree_without_host_objects(config: ExperimentConfig) -> None:
    core = compile_config(config).core
    leaves, structure = jax.tree.flatten(core)
    rebuilt = jax.tree.unflatten(structure, leaves)

    assert isinstance(rebuilt, CoreConfig)
    assert not any(isinstance(leaf, (BaseModel, Path, dict, list)) for leaf in leaves)


def test_filter_jit_consumes_static_and_dynamic_fields(config: ExperimentConfig) -> None:
    @eqx.filter_jit
    def consume_config(core: CoreConfig) -> jax.Array:
        return core.energy.basal_cost + jnp.asarray(core.world.width, dtype=jnp.float32)

    changed = replace(config, "energy", basal_cost=0.25)
    first = consume_config(compile_config(config).core)
    second = consume_config(compile_config(changed).core)

    assert first.shape == second.shape == ()
    assert float(second - first) == pytest.approx(0.15, abs=2e-6)


def test_compiled_contract_is_immutable(config: ExperimentConfig) -> None:
    compiled = compile_config(config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        compiled.config_hash = "changed"
    with pytest.raises(dataclasses.FrozenInstanceError):
        compiled.compile_signature.backend = "auto"


def test_compilation_rejects_unrepresentable_values(config: ExperimentConfig) -> None:
    too_large = replace(config, "world", resource_capacity=1e100)
    with pytest.raises(ConfigCompilationError, match=r"world\.resource_capacity"):
        compile_config(too_large)
