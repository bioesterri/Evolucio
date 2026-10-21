# ruff: noqa: ANN001, ANN201
from dataclasses import fields
from uuid import uuid4

import equinox as eqx
import jax
import numpy as np
import pytest

from evolucio.config.compile import compile_signature_digest
from evolucio.core.metrics import create_empty_metrics_accumulator
from evolucio.persistence.local import verify_local_run
from evolucio.persistence.parquet import ParquetPersistencePolicy
from evolucio.persistence.parquet.reader import scan_dataset
from evolucio.persistence.zarr import load_checkpoint, load_snapshot
from evolucio.persistence.zarr.checkpoint import write_checkpoint
from evolucio.persistence.zarr.index import load_index
from evolucio.persistence.zarr.schemas import ZarrCapturePolicy
from evolucio.runtime import RunPlan
from evolucio.runtime.recorded_run import run_recorded_simulation
from tests.regression.helpers import assert_pytree_equivalent
from tests.zarr_helpers import checkpoint_host_config, checkpoint_scenario


def test_capture_steps_and_logical_states_are_chunk_neutral(tmp_path):
    config = checkpoint_host_config()
    state, prepared = checkpoint_scenario()
    record = write_checkpoint(
        run_dir=tmp_path / "source",
        run_id=str(uuid4()),
        state=state,
        accumulator=create_empty_metrics_accumulator(),
        config=prepared.core,
        config_hash=prepared.config_hash,
        compile_signature=compile_signature_digest(prepared.compile_signature),
    )
    source = tmp_path / "source" / record.relative_path
    policy = ZarrCapturePolicy(
        checkpoint_interval=10,
        snapshot_interval=6,
        checkpoint_on_completion=True,
        snapshot_on_completion=True,
    )
    runs = [
        run_recorded_simulation(
            config=config,
            plan=RunPlan(30, size),
            capture_policy=policy,
            parquet_policy=ParquetPersistencePolicy(metrics_interval=5),
            source_checkpoint=source,
            output_root=tmp_path / "runs",
        )
        for size in (4, 7)
    ]
    indexes = [load_index(r.run_dir) for r in runs]
    for run, index in zip(runs, indexes, strict=True):
        assert [r.state_step for r in index.checkpoints] == [10, 20, 30]
        assert [r.state_step for r in index.snapshots] == [6, 12, 18, 24, 30]
        assert verify_local_run(run.run_dir).ok
    assert_pytree_equivalent(runs[0].result.runtime.state, runs[1].result.runtime.state)
    assert_pytree_equivalent(runs[0].result.runtime.accumulator, runs[1].result.runtime.accumulator)
    for i in range(3):
        states = [
            load_checkpoint(
                checkpoint_path=r.run_dir / indexes[j].checkpoints[i].relative_path
            ).state
            for j, r in enumerate(runs)
        ]
        assert_pytree_equivalent(states[0], states[1])
    for i in range(5):
        snapshots = [
            load_snapshot(r.run_dir / indexes[j].snapshots[i].relative_path)
            for j, r in enumerate(runs)
        ]
        assert snapshots[0].state_step == snapshots[1].state_step
        for component in ("population", "world"):
            left, right = getattr(snapshots[0], component), getattr(snapshots[1], component)
            for field in fields(left):
                a, b = getattr(left, field.name), getattr(right, field.name)
                if isinstance(a, np.ndarray):
                    assert a.dtype == b.dtype
                    if a.dtype.kind == "f":
                        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
                    else:
                        np.testing.assert_array_equal(a, b)
                else:
                    assert a == b
    for run in runs:
        assert scan_dataset(run.run_dir, "metrics").collect()["step"].to_list() == [
            0,
            5,
            10,
            15,
            20,
            25,
        ]
    unrecorded = run_recorded_simulation(
        config=config, plan=RunPlan(30, 7), output_root=tmp_path / "runs", source_checkpoint=source
    )
    assert_pytree_equivalent(runs[1].result.runtime.state, unrecorded.result.runtime.state)
    assert_pytree_equivalent(
        runs[1].result.runtime.accumulator, unrecorded.result.runtime.accumulator
    )
    for name in ("births", "deaths", "genealogy"):
        assert (
            scan_dataset(runs[0].run_dir, name)
            .drop("run_id")
            .collect()
            .equals(scan_dataset(runs[1].run_dir, name).drop("run_id").collect())
        )


@pytest.fixture(scope="module", autouse=True)
def release_test_compilation_caches():
    """Release scenario-specific executables before the rest of the suite runs."""
    yield
    jax.clear_caches()
    eqx.clear_caches()
