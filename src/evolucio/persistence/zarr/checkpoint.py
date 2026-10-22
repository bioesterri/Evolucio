"""Explicit complete carry serialization and validated reconstruction of real PyTrees."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from evolucio import __version__
from evolucio.core.diagnostics import INVARIANT_SCHEMA_DIGEST, check_structural_invariants
from evolucio.core.ids import IdCounters
from evolucio.core.metrics import METRICS_SCHEMA_DIGEST, MetricsAccumulator
from evolucio.core.policy import GENOME_SCHEMA_DIGEST
from evolucio.core.policy.batch import GenomeBatch
from evolucio.core.policy.model import PolicyLinear
from evolucio.core.rng import PRNG_IMPLEMENTATION, RngState
from evolucio.core.simulation import SIMULATION_STEP_SCHEMA_DIGEST
from evolucio.core.state import PopulationState, SimulationState, WorldState
from evolucio.host.dto import HostArray
from evolucio.persistence.local.artifacts import artifact_path
from evolucio.persistence.local.manifest import utc_now
from evolucio.persistence.local.version_info import code_version_info

from .index import CheckpointRecord, capture_path, indexed_record_for_store
from .schemas import (
    ACCUMULATOR_FIELDS,
    CHECKPOINT_SCHEMA_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    ID_FIELDS,
    POPULATION_FIELDS,
    WORLD_FIELDS,
    CheckpointMetadata,
    checkpoint_array_specs,
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

if TYPE_CHECKING:
    from evolucio.config.compile import CoreConfig


def core_schemas() -> dict[str, str]:
    return {
        "simulation_step": SIMULATION_STEP_SCHEMA_DIGEST,
        "metrics": METRICS_SCHEMA_DIGEST,
        "invariants": INVARIANT_SCHEMA_DIGEST,
        "genomes": GENOME_SCHEMA_DIGEST,
    }


def _check_contract() -> None:
    for cls, names in (
        (SimulationState, ("step", "rng", "ids", "world", "population", "genomes")),
        (WorldState, WORLD_FIELDS),
        (PopulationState, POPULATION_FIELDS),
        (IdCounters, ID_FIELDS),
        (MetricsAccumulator, ACCUMULATOR_FIELDS),
        (RngState, ("key",)),
        (GenomeBatch, ("layer1", "layer2")),
        (PolicyLinear, ("weight", "bias")),
    ):
        if tuple(field.name for field in fields(cls)) != names:
            raise CheckpointCompatibilityError(f"checkpoint schema must change for {cls.__name__}")


def _validate_state(
    state: SimulationState, accumulator: MetricsAccumulator, metadata: CheckpointMetadata
) -> None:
    report = check_structural_invariants(
        state=state,
        resource_capacity=jnp.asarray(metadata.resource_capacity, dtype=jnp.float32),  # pyright: ignore[reportUnknownMemberType]
        width=metadata.width,
        height=metadata.height,
    )
    if not bool(report.ok):
        raise CheckpointIntegrityError(
            f"checkpoint state invariants failed: {report.violation_counts}"
        )
    for field in fields(accumulator):
        value = np.asarray(jax.device_get(getattr(accumulator, field.name)))
        if not bool(np.all(np.isfinite(value))) or (
            value.dtype.kind in "iu" and bool(np.any(value < 0))
        ):
            raise CheckpointIntegrityError(f"invalid accumulator: {field.name}")


def read_checkpoint_arrays(path: Path) -> tuple[CheckpointMetadata, dict[str, HostArray]]:
    try:
        raw = read_metadata(path)
        validate_capture_header(
            raw,
            name=CHECKPOINT_SCHEMA_NAME,
            version=CHECKPOINT_SCHEMA_VERSION,
            completion_field="checkpoint_complete",
        )
        metadata = CheckpointMetadata.model_validate_json(json.dumps(raw))
    except ValueError as exc:
        raise CheckpointCompatibilityError(f"unsupported checkpoint metadata: {exc}") from exc
    _check_contract()
    if (
        metadata.core_schemas != core_schemas()
        or metadata.rng_implementation != PRNG_IMPLEMENTATION
    ):
        raise CheckpointCompatibilityError("incompatible core schemas or RNG implementation")
    _, arrays = read_arrays(
        path, checkpoint_array_specs(metadata.width, metadata.height, metadata.capacity)
    )
    if int(arrays["step"]) != metadata.state_step:
        raise CheckpointIntegrityError("state_step does not match the stored step")
    return metadata, arrays


def write_checkpoint(
    *,
    run_dir: Path,
    run_id: str,
    state: SimulationState,
    accumulator: MetricsAccumulator,
    config: CoreConfig,
    config_hash: str,
    compile_signature: str,
) -> CheckpointRecord:
    _check_contract()
    if (
        not jnp.issubdtype(state.rng.key.dtype, jax.dtypes.prng_key)
        or state.rng.key.shape != ()
        or str(jax.random.key_impl(state.rng.key)) != PRNG_IMPLEMENTATION
    ):
        raise CheckpointCompatibilityError("checkpoint requires the canonical scalar typed RNG key")
    info = code_version_info()
    metadata = CheckpointMetadata(
        run_id=run_id,
        state_step=int(state.step),
        config_hash=config_hash,
        compile_signature=compile_signature,
        created_at_utc=utc_now(),
        package_version=__version__,
        git_commit=info.git_commit,
        width=config.world.width,
        height=config.world.height,
        capacity=config.population.max_agents,
        rng_implementation=PRNG_IMPLEMENTATION,
        core_schemas=core_schemas(),
        resource_capacity=float(config.world.resource_capacity),
    )
    _validate_state(state, accumulator, metadata)
    values: dict[str, object] = {
        "step": state.step,
        "rng/key_data": jax.random.key_data(state.rng.key),
    }
    for prefix, obj, names in (
        ("world", state.world, WORLD_FIELDS),
        ("population", state.population, POPULATION_FIELDS),
        ("ids", state.ids, ID_FIELDS),
        ("accumulator", accumulator, ACCUMULATOR_FIELDS),
    ):
        values.update({f"{prefix}/{name}": getattr(obj, name) for name in names})
    for layer in ("layer1", "layer2"):
        for leaf in ("weight", "bias"):
            values[f"genomes/{layer}_{leaf}"] = getattr(getattr(state.genomes, layer), leaf)
    arrays = {name: np.asarray(value) for name, value in jax.device_get(values).items()}
    specs = checkpoint_array_specs(metadata.width, metadata.height, metadata.capacity)
    if any(
        array.shape != specs[name][0] or array.dtype != np.dtype(specs[name][1])
        for name, array in arrays.items()
    ):
        raise CheckpointCompatibilityError("input carry has incompatible shapes or dtypes")
    relative = capture_path("checkpoint", metadata.state_step)
    size, digest = write_store(
        artifact_path(run_dir, relative), arrays, metadata.model_dump(), read_checkpoint_arrays
    )
    return CheckpointRecord(
        state_step=metadata.state_step,
        relative_path=relative,
        byte_size=size,
        sha256=digest,
        created_at_utc=metadata.created_at_utc,
    )


@dataclass(frozen=True, slots=True)
class CheckpointRestore:
    state: SimulationState
    accumulator: MetricsAccumulator
    run_id: str
    state_step: int
    config_hash: str
    compile_signature: str
    checkpoint_sha256: str
    warnings: tuple[str, ...] = ()


def load_checkpoint(
    *,
    checkpoint_path: Path,
    expected_config_hash: str | None = None,
    expected_compile_signature: str | None = None,
) -> CheckpointRestore:
    size, digest = verify_store_tree(checkpoint_path)
    try:
        indexed = indexed_record_for_store(checkpoint_path)
        if indexed is not None and (indexed.sha256 != digest or indexed.byte_size != size):
            raise CheckpointIntegrityError("checkpoint differs from its indexed SHA-256")
    except (OSError, ValueError) as exc:
        raise CheckpointIntegrityError(str(exc)) from exc
    metadata, arrays = read_checkpoint_arrays(checkpoint_path)
    if expected_config_hash is not None and metadata.config_hash != expected_config_hash:
        raise CheckpointCompatibilityError("checkpoint config_hash mismatch")
    if (
        expected_compile_signature is not None
        and metadata.compile_signature != expected_compile_signature
    ):
        raise CheckpointCompatibilityError("checkpoint CompileSignature mismatch")
    device = {name: jnp.asarray(value) for name, value in arrays.items()}  # pyright: ignore[reportUnknownMemberType]

    def component(prefix: str, names: tuple[str, ...]) -> dict[str, jax.Array]:
        return {name: device[f"{prefix}/{name}"] for name in names}

    state = SimulationState(
        step=device["step"],
        rng=RngState(
            key=cast(
                jax.Array,
                jax.random.wrap_key_data(device["rng/key_data"], impl=metadata.rng_implementation),
            )
        ),
        ids=IdCounters(**component("ids", ID_FIELDS)),
        world=WorldState(**component("world", WORLD_FIELDS)),
        population=PopulationState(**component("population", POPULATION_FIELDS)),
        genomes=GenomeBatch(
            layer1=PolicyLinear(
                weight=device["genomes/layer1_weight"], bias=device["genomes/layer1_bias"]
            ),
            layer2=PolicyLinear(
                weight=device["genomes/layer2_weight"], bias=device["genomes/layer2_bias"]
            ),
        ),
    )
    accumulator = MetricsAccumulator(**component("accumulator", ACCUMULATOR_FIELDS))
    _validate_state(state, accumulator, metadata)
    warnings: list[str] = []
    current = code_version_info()
    if metadata.package_version != __version__:
        warnings.append(
            "package version differs; compatibility relies on the explicit schema contracts"
        )
    if metadata.git_commit != current.git_commit:
        warnings.append("Git commit differs; compatibility relies on the explicit schema contracts")
    return CheckpointRestore(
        state,
        accumulator,
        metadata.run_id,
        metadata.state_step,
        metadata.config_hash,
        metadata.compile_signature,
        digest,
        tuple(warnings),
    )
