"""Indexed immutable captures, atomic metadata and checkpoint-only retention."""

import json
import shutil
from pathlib import Path
from typing import Literal

from pydantic import Field

from evolucio.persistence.local.artifacts import artifact_path, atomic_write_text, json_text
from evolucio.persistence.local.recorder import LocalRunRecorder

from .schemas import DEFAULT_ZARR_POLICY, StrictModel, ZarrCapturePolicy

INDEX_PATH = "artifacts/zarr/index.json"


class CaptureRecord(StrictModel):
    state_step: int = Field(ge=0)
    relative_path: str
    byte_size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at_utc: str


class CheckpointRecord(CaptureRecord):
    kind: Literal["checkpoint"] = "checkpoint"
    schema_version: Literal[1] = 1


class SnapshotRecord(CaptureRecord):
    kind: Literal["snapshot"] = "snapshot"
    schema_version: Literal[1] = 1


class ZarrIndex(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str
    config_hash: str
    compile_signature: str
    policy: ZarrCapturePolicy
    complete: bool = False
    checkpoints: list[CheckpointRecord] = Field(default_factory=lambda: list[CheckpointRecord]())
    snapshots: list[SnapshotRecord] = Field(default_factory=lambda: list[SnapshotRecord]())


def capture_path(kind: str, step: int) -> str:
    if kind not in ("checkpoint", "snapshot") or type(step) is not int or step < 0:
        raise ValueError("invalid capture identity")
    return f"artifacts/zarr/{kind}s/step-{step:09d}.zarr"


def load_index(run_dir: Path) -> ZarrIndex:
    text = artifact_path(run_dir, INDEX_PATH).read_text()
    index = ZarrIndex.model_validate_json(text)
    raw = json.loads(text)
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("unsupported or missing Zarr index schema_version")
    if any(
        type(record.get("schema_version")) is not int
        for record in (*raw.get("checkpoints", []), *raw.get("snapshots", []))
    ):
        raise ValueError("capture record requires an explicit schema_version")
    for kind, records in (("checkpoint", index.checkpoints), ("snapshot", index.snapshots)):
        steps: list[int] = []
        for record in records:
            if record.relative_path != capture_path(kind, record.state_step):
                raise ValueError("invalid capture path")
            artifact_path(run_dir, record.relative_path)
            steps.append(record.state_step)
        if steps != sorted(set(steps)):
            raise ValueError("capture steps must be ordered and unique")
    return index


def save_index(run_dir: Path, index: ZarrIndex) -> None:
    atomic_write_text(artifact_path(run_dir, INDEX_PATH), json_text(index.model_dump()))


def latest_checkpoint(run_dir: Path) -> CheckpointRecord | None:
    if not artifact_path(run_dir, INDEX_PATH).exists():
        return None
    records = load_index(run_dir).checkpoints
    return records[-1] if records else None


def list_snapshots(run_dir: Path) -> tuple[SnapshotRecord, ...]:
    if not artifact_path(run_dir, INDEX_PATH).exists():
        return ()
    return tuple(load_index(run_dir).snapshots)


class ZarrRunIndex:
    """Single writer; a new checkpoint is indexed before any older store is removed."""

    def __init__(
        self, recorder: LocalRunRecorder, policy: ZarrCapturePolicy = DEFAULT_ZARR_POLICY
    ) -> None:
        if recorder.manifest.compile_signature is None:
            raise ValueError("Zarr captures require the run compilation signature")
        self.recorder = recorder
        self.closed = False
        self.index = ZarrIndex(
            run_id=recorder.run_id,
            config_hash=recorder.manifest.config_hash,
            compile_signature=recorder.manifest.compile_signature,
            policy=policy,
        )
        artifact_path(recorder.run_dir, "artifacts/zarr").mkdir(exist_ok=False)
        save_index(recorder.run_dir, self.index)

    def add(self, record: CheckpointRecord | SnapshotRecord) -> None:
        if self.closed:
            raise ValueError("Zarr index is finalized")
        if record.relative_path != capture_path(record.kind, record.state_step):
            raise ValueError("capture record has an invalid relative path")
        artifact_path(self.recorder.run_dir, record.relative_path)
        previous = (
            self.index.checkpoints if isinstance(record, CheckpointRecord) else self.index.snapshots
        )
        if previous and record.state_step <= previous[-1].state_step:
            raise ValueError("capture records must advance their absolute state step")
        candidate = self.index.model_copy(deep=True)
        if isinstance(record, CheckpointRecord):
            candidate.checkpoints.append(record)
        else:
            candidate.snapshots.append(record)
        save_index(self.recorder.run_dir, candidate)
        self.index = candidate
        keep = candidate.policy.keep_last_checkpoints
        if (
            isinstance(record, CheckpointRecord)
            and keep is not None
            and len(candidate.checkpoints) > keep
        ):
            old = candidate.checkpoints[:-keep]
            for entry in old:
                shutil.rmtree(artifact_path(self.recorder.run_dir, entry.relative_path))
            candidate = candidate.model_copy(deep=True)
            candidate.checkpoints = candidate.checkpoints[-keep:]
            save_index(self.recorder.run_dir, candidate)
            self.index = candidate

    def finalize(self, *, complete: bool = True) -> None:
        if self.closed:
            raise ValueError("Zarr index already finalized")
        if complete:
            from .integrity import verify_zarr_artifacts

            report = verify_zarr_artifacts(self.recorder.run_dir)
            if not report.ok:
                raise ValueError(f"Zarr integrity verification failed: {report}")
        self.index.complete = complete
        save_index(self.recorder.run_dir, self.index)
        self.recorder.register_artifact(
            name="zarr_artifact_index",
            kind="zarr_artifact_index",
            relative_path=INDEX_PATH,
            media_type="application/json",
        )
        self.closed = True


def indexed_record_for_store(path: Path) -> CheckpointRecord | SnapshotRecord | None:
    """Return only a matching indexed identity; never adopt a filesystem orphan."""
    resolved = path.resolve()
    if len(resolved.parents) < 4:
        return None
    run_dir = resolved.parents[3]
    if not artifact_path(run_dir, INDEX_PATH).is_file():
        return None
    relative = resolved.relative_to(run_dir).as_posix()
    index = load_index(run_dir)
    return next(
        (
            record
            for record in (*index.checkpoints, *index.snapshots)
            if record.relative_path == relative
        ),
        None,
    )
