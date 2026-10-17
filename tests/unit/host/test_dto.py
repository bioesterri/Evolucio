# ruff: noqa: ANN201

import subprocess
import sys
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from evolucio.host import EventBatchDTO, SimulationSnapshotDTO, WorldSnapshotDTO


def test_import_is_independent_of_device_and_core():
    script = """
import sys
import importlib.abc
class BlockDevice(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('jax', 'equinox', 'evolucio.core', 'evolucio.runtime'):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, BlockDevice())
import evolucio.host
for name in ('jax', 'equinox', 'evolucio.core', 'evolucio.runtime'):
    assert name not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_frozen_slotted_scalar_contract():
    dto = EventBatchDTO(births=(), deaths=())
    assert not hasattr(dto, "__dict__")
    with pytest.raises(FrozenInstanceError):
        dto.births = ()


def test_world_owns_read_only_dtype_preserving_arrays():
    original = np.arange(6, dtype=np.float32).reshape(2, 3)
    dto = WorldSnapshotDTO(
        width=3,
        height=2,
        resources=original,
        environment=original,
        occupancy=np.zeros((2, 3), np.int32),
    )
    original[:] = 90
    np.testing.assert_array_equal(dto.resources, np.arange(6, dtype=np.float32).reshape(2, 3))
    for name in ("resources", "environment", "occupancy"):
        array = getattr(dto, name)
        assert array.flags.owndata
        assert not array.flags.writeable
        assert not np.shares_memory(array, original)
        with pytest.raises(ValueError):
            array[0, 0] = 0
    assert dto.resources.dtype == np.float32
    assert dto.occupancy.dtype == np.int32


def test_snapshot_is_not_a_checkpoint():
    names = {field.name for field in fields(SimulationSnapshotDTO)}
    assert names == {
        "schema_version",
        "state_step",
        "next_agent_id",
        "next_genome_id",
        "next_lineage_id",
        "population",
        "world",
    }
    assert names.isdisjoint({"rng", "genomes", "accumulator", "config", "events", "run_id", "step"})
    assert not hasattr(SimulationSnapshotDTO, "to_simulation_state")


def test_snapshot_constructor_rejects_non_numpy_storage():
    values = np.zeros((1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="NumPy host array"):
        WorldSnapshotDTO(width=1, height=1, resources=[[0.0]], environment=values, occupancy=values)
