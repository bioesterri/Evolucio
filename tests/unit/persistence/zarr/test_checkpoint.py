# ruff: noqa: ANN001, ANN201
from dataclasses import fields

import numpy as np
import pytest
import zarr

from evolucio.core.metrics import MetricsAccumulator
from evolucio.persistence.zarr.checkpoint import load_checkpoint, write_checkpoint
from evolucio.persistence.zarr.schemas import (
    ACCUMULATOR_FIELDS,
    CHECKPOINT_SCHEMA_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    checkpoint_array_specs,
)
from evolucio.persistence.zarr.storage import (
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    read_arrays,
    read_metadata,
    seal_store,
)
from tests.regression.helpers import assert_pytree_equivalent


def test_complete_carry_roundtrip(checkpoint, carry):
    state, accumulator, config, digest = carry
    restored = load_checkpoint(
        checkpoint_path=checkpoint, expected_config_hash=digest, expected_compile_signature="a" * 64
    )
    assert_pytree_equivalent(restored.state, state, float_atol=0, float_rtol=0)
    assert_pytree_equivalent(restored.accumulator, accumulator, float_atol=0, float_rtol=0)
    assert restored.state.rng.key.dtype == state.rng.key.dtype
    assert tuple(f.name for f in fields(MetricsAccumulator)) == ACCUMULATOR_FIELDS
    metadata = read_metadata(checkpoint)
    assert metadata["checkpoint_complete"] is True
    assert CHECKPOINT_SCHEMA_VERSION == SNAPSHOT_SCHEMA_VERSION == 1
    _, arrays = read_arrays(checkpoint, checkpoint_array_specs(6, 6, 8))
    assert len(arrays) == 41
    for name, array in arrays.items():
        if name.startswith("genomes/"):
            assert np.all(array[3:] == 0)
    assert config.population.max_agents == restored.state.population.alive.shape[0]
    with pytest.raises(FileExistsError):
        write_checkpoint(
            run_dir=checkpoint.parents[3],
            run_id=restored.run_id,
            state=state,
            accumulator=accumulator,
            config=config,
            config_hash=digest,
            compile_signature="a" * 64,
        )


@pytest.mark.parametrize(
    "kwargs", [{"expected_config_hash": "b" * 64}, {"expected_compile_signature": "b" * 64}]
)
def test_reject_context_mismatch(checkpoint, kwargs):
    with pytest.raises(CheckpointCompatibilityError):
        load_checkpoint(checkpoint_path=checkpoint, **kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("rng_implementation", "rbg"),
        ("core_schemas", {}),
        ("schema_name", None),
        ("schema_version", None),
        ("checkpoint_complete", None),
    ],
)
def test_reject_incompatible_metadata(checkpoint, field, value):
    group = zarr.open_group(str(checkpoint), mode="a")
    if value is None:
        del group.attrs[field]
    else:
        group.attrs[field] = value
    seal_store(checkpoint)
    with pytest.raises(CheckpointCompatibilityError):
        load_checkpoint(checkpoint_path=checkpoint)


def test_structural_invariants_not_repaired(checkpoint):
    group = zarr.open_group(str(checkpoint), mode="a")
    group["population/agent_id"][0] = -1
    seal_store(checkpoint)
    with pytest.raises(CheckpointIntegrityError, match="invariants"):
        load_checkpoint(checkpoint_path=checkpoint)
    assert group["population/agent_id"][0] == -1


def test_wrong_shape_or_dtype(checkpoint):
    group = zarr.open_group(str(checkpoint), mode="a")
    del group["world/resources"]
    group.create_array("world/resources", data=np.zeros((2, 3), np.float64))
    seal_store(checkpoint)
    with pytest.raises(CheckpointCompatibilityError, match="shape/dtype"):
        load_checkpoint(checkpoint_path=checkpoint)


def test_signed_accumulator_and_version_warning(tmp_path, carry, monkeypatch):
    import equinox as eqx
    import jax.numpy as jnp

    from evolucio.persistence.local.version_info import CodeVersionInfo

    state, accumulator, config, digest = carry
    accumulator = eqx.tree_at(
        lambda a: a.death_energy_removed_total, accumulator, jnp.asarray(-0.25, jnp.float32)
    )
    record = write_checkpoint(
        run_dir=tmp_path,
        run_id="origin",
        state=state,
        accumulator=accumulator,
        config=config,
        config_hash=digest,
        compile_signature="a" * 64,
    )
    monkeypatch.setattr(
        "evolucio.persistence.zarr.checkpoint.code_version_info",
        lambda: CodeVersionInfo("different", "b" * 40, False),
    )
    loaded = load_checkpoint(checkpoint_path=tmp_path / record.relative_path)
    assert float(loaded.accumulator.death_energy_removed_total) == -0.25
    assert loaded.warnings and "Git commit differs" in loaded.warnings[-1]


def test_nonfinite_accumulator_rejected(tmp_path, carry):
    import equinox as eqx
    import jax.numpy as jnp

    state, accumulator, config, digest = carry
    accumulator = eqx.tree_at(
        lambda a: a.resource_consumed_total, accumulator, jnp.asarray(float("nan"), jnp.float32)
    )
    with pytest.raises(CheckpointIntegrityError, match="accumulator"):
        write_checkpoint(
            run_dir=tmp_path,
            run_id="origin",
            state=state,
            accumulator=accumulator,
            config=config,
            config_hash=digest,
            compile_signature="a" * 64,
        )


def test_schema_does_not_silently_drop_future_fields(carry, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from evolucio.core.state import SimulationState
    from evolucio.persistence.zarr import checkpoint as api

    original = api.fields
    monkeypatch.setattr(
        api,
        "fields",
        lambda cls: (
            (*original(cls), SimpleNamespace(name="new_leaf"))
            if cls is SimulationState
            else original(cls)
        ),
    )
    state, accumulator, config, digest = carry
    with pytest.raises(CheckpointCompatibilityError, match="schema must change"):
        write_checkpoint(
            run_dir=tmp_path,
            run_id="origin",
            state=state,
            accumulator=accumulator,
            config=config,
            config_hash=digest,
            compile_signature="a" * 64,
        )


def test_legacy_untyped_rng_is_not_silently_reinterpreted(tmp_path, carry):
    import equinox as eqx
    import jax

    state, accumulator, config, digest = carry
    state = eqx.tree_at(lambda s: s.rng.key, state, jax.random.PRNGKey(1729))
    with pytest.raises(CheckpointCompatibilityError, match="typed RNG"):
        write_checkpoint(
            run_dir=tmp_path,
            run_id="origin",
            state=state,
            accumulator=accumulator,
            config=config,
            config_hash=digest,
            compile_signature="a" * 64,
        )
