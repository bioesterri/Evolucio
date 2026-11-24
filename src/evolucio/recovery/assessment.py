"""Read-only validation of indexed checkpoints and their experimental prefixes."""

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import polars as pl

from evolucio.config import ExperimentConfig, freeze_config, load_config
from evolucio.config.compile import build_compile_signature, compile_signature_digest
from evolucio.config.errors import ConfigError
from evolucio.config.versions import CONFIG_SCHEMA_VERSION
from evolucio.host.schema import HOST_DTO_SCHEMA_DIGEST
from evolucio.persistence.local.artifacts import artifact_path, sha256_file
from evolucio.persistence.local.manifest import RunManifest, RunStatus, load_run_manifest
from evolucio.persistence.local.ownership import WriterStatus, run_ownership
from evolucio.persistence.local.version_info import code_version_info, environment_info
from evolucio.persistence.parquet.index import load_index as load_telemetry
from evolucio.persistence.parquet.index import prefix_digest
from evolucio.persistence.zarr.checkpoint import CheckpointRestore, load_checkpoint
from evolucio.persistence.zarr.index import CheckpointRecord, load_index
from evolucio.runtime.modes import RunMode

from .history import prefix_frames, verify_history
from .provenance import read_provenance


@dataclass(frozen=True, slots=True)
class RecoveryAssessment:
    source_run_id: str
    source_status: RunStatus
    writer_active: WriterStatus
    config_hash: str
    target_step: int
    last_confirmed_state_step: int | None
    last_committed_telemetry_step: int | None
    available_checkpoints: tuple[int, ...]
    latest_valid_checkpoint_step: int | None
    latest_recoverable_checkpoint_step: int | None
    recoverable_steps: tuple[int, ...]
    incompatibilities: tuple[str, ...]
    integrity_errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def recoverable(self) -> bool:
        return bool(self.recoverable_steps)


def restore_indexed(
    run_dir: Path, manifest: RunManifest, record: CheckpointRecord
) -> CheckpointRestore:
    restored = load_checkpoint(
        checkpoint_path=artifact_path(run_dir, record.relative_path),
        expected_config_hash=manifest.config_hash,
        expected_compile_signature=manifest.compile_signature,
    )
    if (restored.run_id, restored.state_step, restored.checkpoint_sha256) != (
        manifest.run_id,
        record.state_step,
        record.sha256,
    ):
        raise ValueError("checkpoint store does not match indexed identity")
    return restored


def load_canonical_config(run_dir: Path, manifest: RunManifest) -> ExperimentConfig:
    path = artifact_path(run_dir, manifest.config_artifact)
    record = next(r for r in manifest.artifacts if r.relative_path == manifest.config_artifact)
    config = load_config(path)
    if (
        sha256_file(path) != record.sha256
        or freeze_config(config).config_hash != manifest.config_hash
        or config.seed != manifest.seed
        or run_dir.name != manifest.run_id
    ):
        raise ValueError("frozen configuration or run identity differs")
    return config


def inspect_run(run_dir: Path) -> RecoveryAssessment:
    with run_ownership(run_dir) as ownership:
        return inspect_owned(run_dir, ownership)


def inspect_owned(run_dir: Path, ownership: WriterStatus) -> RecoveryAssessment:
    manifest = load_run_manifest(run_dir / "manifest.json")
    incompatibilities: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    target = manifest.initial_step + manifest.total_steps
    if ownership != "inactive":
        incompatibilities.append(f"source writer is {ownership}")
    if manifest.status is RunStatus.RUNNING and ownership == "inactive":
        warnings.append("run has no writer and requires explicit lifecycle reconciliation")
    elif manifest.status not in (RunStatus.FAILED, RunStatus.STOPPED):
        incompatibilities.append("source must be failed or stopped")
    if manifest.mode is not RunMode.EXPERIMENT:
        incompatibilities.append("only experiment runs can be recovered")
    if manifest.initial_step:
        try:
            if read_provenance(run_dir, manifest) is None:
                incompatibilities.append("original trajectory provenance is unknown")
        except (OSError, ValueError) as exc:
            incompatibilities.append(str(exc))
    code = code_version_info()
    env = environment_info(backend=jax.default_backend())
    if code.git_commit is None or code.git_dirty is not False or code != manifest.code_version:
        incompatibilities.append("clean, matching Git and package identity required")
    if env != manifest.environment or None in (
        env.jax_version,
        env.equinox_version,
        env.jaxlib_version,
    ):
        incompatibilities.append("computational environment differs or is unknown")
    signature = None
    try:
        config = load_canonical_config(run_dir, manifest)
        signature = build_compile_signature(config)
        if compile_signature_digest(signature) != manifest.compile_signature:
            incompatibilities.append("CompileSignature differs")
        if (
            manifest.schemas.config != CONFIG_SCHEMA_VERSION
            or manifest.schemas.host_dto != HOST_DTO_SCHEMA_DIGEST
            or manifest.schemas.simulation_step != signature.simulation_step_schema_digest
            or manifest.schemas.metrics != signature.metrics_schema_digest
        ):
            incompatibilities.append("manifest scientific schemas differ")
    except (OSError, ValueError, ConfigError) as exc:
        incompatibilities.append(str(exc))
    records: list[CheckpointRecord] = []
    indexed: set[str] = set()
    telemetry = None
    try:
        zarr_index = load_index(run_dir)
        if (zarr_index.run_id, zarr_index.config_hash, zarr_index.compile_signature) != (
            manifest.run_id,
            manifest.config_hash,
            manifest.compile_signature,
        ):
            raise ValueError("Zarr index identity differs from manifest")
        records = list(reversed(zarr_index.checkpoints))
        indexed.update(record.relative_path for record in (*records, *zarr_index.snapshots))
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    try:
        telemetry = load_telemetry(run_dir)
        if telemetry.run_id != manifest.run_id:
            raise ValueError("Parquet index identity differs from manifest")
        indexed.update(
            part.relative_path for ds in telemetry.datasets.values() for part in ds.parts
        )
    except (OSError, ValueError) as exc:
        telemetry = None
        errors.append(str(exc))
    tracking_path = artifact_path(run_dir, "artifacts/mlflow/tracking.json")
    if telemetry is not None and tracking_path.is_file():
        try:
            interval = json.loads(tracking_path.read_bytes())["metrics_interval"]
            if type(interval) is not int or interval <= 0:
                raise ValueError("invalid MLflow metrics interval")
            if interval % telemetry.policy.metrics_interval:
                warnings.append("MLflow sampling will use only steps shared with canonical Parquet")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"invalid MLflow tracking metadata: {exc}")
    for path in sorted((run_dir / "artifacts").rglob("*")):
        relative = path.relative_to(run_dir).as_posix()
        if path.name.endswith(".tmp") or ".tmp-" in path.name:
            warnings.append(f"temporary artifact: {relative}")
        elif (
            path.suffix == ".zarr" or (path.name.startswith("part-") and path.suffix == ".parquet")
        ) and relative not in indexed:
            warnings.append(f"unindexed artifact: {relative}")
    valid: list[int] = []
    recoverable: list[int] = []
    for record in records:
        step = record.state_step
        try:
            restored = restore_indexed(run_dir, manifest, record)
            if signature is not None and (
                restored.state.world.resources.shape
                != (signature.world_height, signature.world_width)
                or restored.state.population.alive.shape != (signature.max_agents,)
            ):
                raise ValueError("checkpoint shapes disagree with the frozen configuration")
            valid.append(step)
            warnings.extend(restored.warnings)
            if step >= target:
                continue
            if telemetry is None:
                raise ValueError("no telemetry index")
            if (
                record.committed_telemetry_end_step != step
                or record.telemetry_prefix_digest != prefix_digest(telemetry, step)
            ):
                raise ValueError("checkpoint lacks a matching committed telemetry prefix")
            verify_history(run_dir, manifest, telemetry, restored)
            if not incompatibilities:
                recoverable.append(step)
        except (OSError, ValueError, KeyError, TypeError, pl.exceptions.PolarsError) as exc:
            errors.append(f"checkpoint {step}: {exc}")
    committed_step: int | None = None
    if telemetry is not None and telemetry.chunks:
        try:
            for chunk, _ in prefix_frames(
                run_dir, telemetry, telemetry.chunks[-1].end_step_exclusive
            ):
                committed_step = chunk.end_step_exclusive
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            errors.append(f"committed telemetry: {exc}")
    terminal = manifest.error or manifest.stop
    return RecoveryAssessment(
        manifest.run_id,
        manifest.status,
        ownership,
        manifest.config_hash,
        target,
        None if terminal is None else terminal.last_confirmed_state_step,
        committed_step,
        tuple(record.state_step for record in records),
        next(iter(valid), None),
        next(iter(recoverable), None),
        tuple(recoverable),
        tuple(incompatibilities),
        tuple(errors),
        tuple(dict.fromkeys(warnings)),
    )
