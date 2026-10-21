# ruff: noqa: ANN001, ANN201
from dataclasses import fields, replace

import numpy as np
import pytest

from evolucio.host import SimulationSnapshotDTO
from evolucio.persistence.zarr import load_snapshot
from evolucio.persistence.zarr.checkpoint import load_checkpoint
from evolucio.persistence.zarr.snapshot import write_snapshot
from evolucio.persistence.zarr.storage import CheckpointCompatibilityError, read_metadata
from evolucio.runtime.snapshot import simulation_state_to_snapshot


def test_host_dto_only_and_dtype_preservation(tmp_path, carry):
    state, _, _, digest = carry
    dto = simulation_state_to_snapshot(state)
    population = replace(dto.population, energy=dto.population.energy.astype(np.float64))
    dto = replace(dto, population=population)
    record = write_snapshot(
        run_dir=tmp_path,
        run_id="snapshot",
        snapshot=dto,
        config_hash=digest,
        compile_signature="a" * 64,
    )
    path = tmp_path / record.relative_path
    loaded = load_snapshot(path)
    assert isinstance(loaded, SimulationSnapshotDTO)
    assert loaded.state_step == dto.state_step
    for component in ("population", "world"):
        original, restored = getattr(dto, component), getattr(loaded, component)
        for field in fields(original):
            a, b = getattr(original, field.name), getattr(restored, field.name)
            if isinstance(a, np.ndarray):
                assert np.array_equal(a, b) and a.dtype == b.dtype
            else:
                assert a == b
    names = [p.relative_to(path).as_posix() for p in path.rglob("zarr.json")]
    assert not any(n.startswith(("rng/", "genomes/", "accumulator/")) for n in names)
    assert "rng_implementation" not in read_metadata(path)
    with pytest.raises(CheckpointCompatibilityError):
        load_checkpoint(checkpoint_path=path)
    import zarr

    from evolucio.persistence.zarr.storage import seal_store

    group = zarr.open_group(str(path), mode="a")
    del group.attrs["snapshot_complete"]
    seal_store(path)
    with pytest.raises(CheckpointCompatibilityError, match="incomplete"):
        load_snapshot(path)


def test_no_snapshot_restore_api():
    import evolucio.persistence.zarr as api

    assert not hasattr(api, "snapshot_to_simulation_state")
