"""Analytical host DTO serialization; intentionally has no checkpoint restore path."""

import json
from pathlib import Path

import numpy as np

from evolucio import __version__
from evolucio.host import (
    HOST_DTO_SCHEMA_VERSION,
    PopulationSnapshotDTO,
    SimulationSnapshotDTO,
    WorldSnapshotDTO,
)
from evolucio.host.dto import HostArray
from evolucio.host.schema import HOST_DTO_SCHEMA_DIGEST
from evolucio.persistence.local.artifacts import artifact_path
from evolucio.persistence.local.manifest import utc_now
from evolucio.persistence.local.version_info import code_version_info

from .index import SnapshotRecord, capture_path, indexed_record_for_store
from .schemas import (
    SNAPSHOT_POPULATION_FIELDS,
    SNAPSHOT_SCHEMA_NAME,
    SNAPSHOT_SCHEMA_VERSION,
    WORLD_FIELDS,
    SnapshotMetadata,
    validate_capture_header,
)
from .storage import (
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    read_arrays,
    read_metadata,
    verify_store_tree,
    write_store,
)


def _specs(metadata: SnapshotMetadata) -> dict[str, tuple[tuple[int, ...], str]]:
    shapes = {
        f"population/{name}": (metadata.alive_count, 2)
        if name == "position"
        else (metadata.alive_count,)
        for name in SNAPSHOT_POPULATION_FIELDS
    }
    shapes.update({f"world/{name}": (metadata.height, metadata.width) for name in WORLD_FIELDS})
    if (
        set(shapes) != set(metadata.array_dtypes)
        or metadata.host_dto_digest != HOST_DTO_SCHEMA_DIGEST
    ):
        raise CheckpointCompatibilityError("incompatible snapshot schema")
    specs: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, shape in shapes.items():
        try:
            dtype = np.dtype(metadata.array_dtypes[name])
        except TypeError as exc:
            raise CheckpointCompatibilityError(f"unsupported snapshot dtype: {name}") from exc
        kinds = (
            "f" if name in ("population/energy", "world/resources", "world/environment") else "iu"
        )
        if dtype.kind not in kinds or dtype.itemsize not in (1, 2, 4, 8):
            raise CheckpointCompatibilityError(f"invalid snapshot dtype: {name}")
        specs[name] = (shape, dtype.str)
    return specs


def _validate(snapshot: SimulationSnapshotDTO) -> None:
    p, w = snapshot.population, snapshot.world
    if snapshot.schema_version != HOST_DTO_SCHEMA_VERSION or p.alive_count > p.capacity:
        raise CheckpointCompatibilityError("incompatible snapshot DTO")
    ids = p.agent_id
    if (
        np.any(ids < 0)
        or np.any(ids[1:] <= ids[:-1])
        or len(np.unique(p.genome_id)) != len(ids)
        or np.any(p.slot_index < 0)
        or np.any(p.slot_index >= p.capacity)
        or len(np.unique(p.slot_index)) != len(ids)
    ):
        raise CheckpointIntegrityError("invalid snapshot identities or canonical order")
    for array, counter in (
        (ids, snapshot.next_agent_id),
        (p.genome_id, snapshot.next_genome_id),
        (p.lineage_id, snapshot.next_lineage_id),
    ):
        if np.any(array < 0) or np.any(array >= counter):
            raise CheckpointIntegrityError("snapshot identity counter mismatch")
    if (
        np.any(p.generation < 0)
        or np.any(p.age < 0)
        or np.any(p.birth_step < 0)
        or np.any(p.birth_step > snapshot.state_step)
        or np.any((p.generation == 0) & (p.parent_id != -1))
        or np.any((p.generation > 0) & ((p.parent_id < 0) | (p.parent_id >= ids)))
    ):
        raise CheckpointIntegrityError("invalid snapshot genealogy or age")
    if (
        np.any(p.position < 0)
        or np.any(p.position[:, 0] >= w.width)
        or np.any(p.position[:, 1] >= w.height)
    ):
        raise CheckpointIntegrityError("snapshot positions are outside the world")
    for array in (p.energy, w.resources, w.environment):
        if not np.all(np.isfinite(array)):
            raise CheckpointIntegrityError("non-finite snapshot values")
    occupancy = np.zeros((w.height, w.width), dtype=np.int64)
    np.add.at(occupancy, (p.position[:, 1], p.position[:, 0]), 1)
    if np.any(w.resources < 0) or not np.array_equal(occupancy, w.occupancy):
        raise CheckpointIntegrityError("snapshot world is inconsistent")


def read_snapshot(path: Path) -> SimulationSnapshotDTO:
    try:
        raw = read_metadata(path)
        validate_capture_header(
            raw,
            name=SNAPSHOT_SCHEMA_NAME,
            version=SNAPSHOT_SCHEMA_VERSION,
            completion_field="snapshot_complete",
        )
        metadata = SnapshotMetadata.model_validate_json(json.dumps(raw))
    except ValueError as exc:
        raise CheckpointCompatibilityError(f"unsupported snapshot metadata: {exc}") from exc
    _, arrays = read_arrays(path, _specs(metadata))
    population = PopulationSnapshotDTO(
        capacity=metadata.capacity,
        alive_count=metadata.alive_count,
        **{name: arrays[f"population/{name}"] for name in SNAPSHOT_POPULATION_FIELDS},
    )
    world = WorldSnapshotDTO(
        width=metadata.width,
        height=metadata.height,
        **{name: arrays[f"world/{name}"] for name in WORLD_FIELDS},
    )
    dto = SimulationSnapshotDTO(
        HOST_DTO_SCHEMA_VERSION,
        metadata.state_step,
        metadata.next_agent_id,
        metadata.next_genome_id,
        metadata.next_lineage_id,
        population,
        world,
    )
    _validate(dto)
    return dto


def write_snapshot(
    *,
    run_dir: Path,
    run_id: str,
    snapshot: SimulationSnapshotDTO,
    config_hash: str,
    compile_signature: str,
) -> SnapshotRecord:
    _validate(snapshot)
    info = code_version_info()
    arrays: dict[str, HostArray] = {
        f"population/{name}": getattr(snapshot.population, name)
        for name in SNAPSHOT_POPULATION_FIELDS
    }
    arrays.update({f"world/{name}": getattr(snapshot.world, name) for name in WORLD_FIELDS})
    metadata = SnapshotMetadata(
        run_id=run_id,
        state_step=snapshot.state_step,
        config_hash=config_hash,
        compile_signature=compile_signature,
        created_at_utc=utc_now(),
        package_version=__version__,
        git_commit=info.git_commit,
        width=snapshot.world.width,
        height=snapshot.world.height,
        capacity=snapshot.population.capacity,
        alive_count=snapshot.population.alive_count,
        next_agent_id=snapshot.next_agent_id,
        next_genome_id=snapshot.next_genome_id,
        next_lineage_id=snapshot.next_lineage_id,
        array_dtypes={name: array.dtype.str for name, array in arrays.items()},
    )
    _specs(metadata)
    relative = capture_path("snapshot", snapshot.state_step)
    size, digest = write_store(
        artifact_path(run_dir, relative), arrays, metadata.model_dump(), read_snapshot
    )
    return SnapshotRecord(
        state_step=snapshot.state_step,
        relative_path=relative,
        byte_size=size,
        sha256=digest,
        created_at_utc=metadata.created_at_utc,
    )


def load_snapshot(path: Path) -> SimulationSnapshotDTO:
    size, digest = verify_store_tree(path)
    indexed = indexed_record_for_store(path)
    if indexed is not None and (indexed.sha256 != digest or indexed.byte_size != size):
        raise CheckpointIntegrityError("snapshot differs from its indexed SHA-256")
    return read_snapshot(path)
