# ruff: noqa: ANN001, ANN201
from uuid import uuid4

import equinox as eqx
import jax
import numpy as np
import pytest

from evolucio.config.compile import compile_signature_digest
from evolucio.core.metrics import create_empty_metrics_accumulator
from evolucio.persistence.zarr.checkpoint import load_checkpoint, write_checkpoint
from evolucio.runtime import DTOChunkConsumer, RunPlan, run_simulation
from tests.regression.helpers import assert_pytree_equivalent
from tests.zarr_helpers import assert_checkpoint_bitwise_equal, checkpoint_scenario


@pytest.fixture(scope="module")
def reference():
    state, prepared = checkpoint_scenario()
    batches = []
    result = run_simulation(
        state=state,
        accumulator=create_empty_metrics_accumulator(),
        config=prepared.core,
        plan=RunPlan(100, 7),
        consumer=DTOChunkConsumer(batches.append),
    )
    assert int(result.accumulator.births_total) > 0
    assert int(result.accumulator.deaths_total) > 0
    assert int(result.accumulator.movement_success_total) > 0
    assert int(result.accumulator.feeding_success_total) > 0
    assert any(b.mutation_effective_count > 0 for batch in batches for b in batch.events.births)
    return state, prepared, result


@pytest.mark.parametrize("cut", [1, 17, 32, 40])
def test_continuous_equals_checkpoint_restore_continuation(tmp_path, reference, cut):
    state, prepared, continuous = reference
    first = run_simulation(
        state=state,
        accumulator=create_empty_metrics_accumulator(),
        config=prepared.core,
        plan=RunPlan(cut, 7),
    )
    signature = compile_signature_digest(prepared.compile_signature)
    record = write_checkpoint(
        run_dir=tmp_path,
        run_id=str(uuid4()),
        state=first.state,
        accumulator=first.accumulator,
        config=prepared.core,
        config_hash=prepared.config_hash,
        compile_signature=signature,
    )
    restored = load_checkpoint(
        checkpoint_path=tmp_path / record.relative_path,
        expected_config_hash=prepared.config_hash,
        expected_compile_signature=signature,
    )
    assert_checkpoint_bitwise_equal(
        (restored.state, restored.accumulator), (first.state, first.accumulator)
    )
    assert_pytree_equivalent(restored.state, first.state, float_atol=0, float_rtol=0)
    assert_pytree_equivalent(restored.accumulator, first.accumulator, float_atol=0, float_rtol=0)
    last = run_simulation(
        state=restored.state,
        accumulator=restored.accumulator,
        config=prepared.core,
        plan=RunPlan(100 - cut, 7),
    )
    assert_pytree_equivalent(last.state, continuous.state)
    assert_pytree_equivalent(last.accumulator, continuous.accumulator)
    np.testing.assert_array_equal(
        jax.random.key_data(last.state.rng.key), jax.random.key_data(continuous.state.rng.key)
    )
    assert_pytree_equivalent(last.state.ids, continuous.state.ids)


@pytest.fixture(scope="module", autouse=True)
def release_test_compilation_caches():
    """Release scenario-specific executables before the rest of the suite runs."""
    yield
    jax.clear_caches()
    eqx.clear_caches()
