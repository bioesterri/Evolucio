"""Immutable, validated lifecycle metadata and strict loading of local manifests."""

# pyright: reportUnnecessaryIsInstance=false

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, cast
from uuid import UUID

from pydantic import ConfigDict, TypeAdapter, ValidationError

from evolucio.runtime.modes import RunMode
from evolucio.runtime.types import RunPlan

from .artifacts import ArtifactRecord, validate_relative_path, validate_sha256
from .schema import RUN_MANIFEST_SCHEMA_NAME, RUN_MANIFEST_SCHEMA_VERSION
from .version_info import CodeVersionInfo, EnvironmentInfo


class ManifestValidationError(ValueError):
    """The file is not a supported, internally consistent local manifest."""


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def validate_run_id(value: str) -> str:
    if str(UUID(value)) != value:
        raise ValueError("run_id must be a canonical UUID string")
    return value


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _timestamp(value: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError("timestamps must be ISO 8601 UTC with a Z suffix")
    return datetime.fromisoformat(value)


@dataclass(frozen=True, slots=True)
class RunErrorRecord:
    error_type: str
    message: str
    step: int | None = None

    def __post_init__(self) -> None:
        if not self.error_type or len(self.message) > 4096:
            raise ValueError("error requires a type and a bounded message")
        if self.step is not None and (isinstance(self.step, bool) or self.step < 0):
            raise ValueError("error step must be nonnegative or unknown")


@dataclass(frozen=True, slots=True)
class RunSchemas:
    config: str
    host_dto: str
    simulation_step: str | None = None
    metrics: str | None = None

    def __post_init__(self) -> None:
        if not self.config:
            raise ValueError("config schema is required")
        validate_sha256(self.host_dto)
        for value in (self.simulation_step, self.metrics):
            if value is not None:
                validate_sha256(value)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Operational context and an immutable artifact index; no simulation carry."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    manifest_schema_name: str
    manifest_schema_version: int
    run_id: str
    status: RunStatus
    created_at_utc: str
    started_at_utc: str | None
    finished_at_utc: str | None
    seed: int
    config_hash: str
    config_artifact: str
    total_steps: int
    chunk_steps: int
    mode: RunMode
    initial_step: int
    final_step: int | None
    steps_executed: int | None
    code_version: CodeVersionInfo
    environment: EnvironmentInfo
    schemas: RunSchemas
    compile_signature: str | None
    artifacts: tuple[ArtifactRecord, ...]
    error: RunErrorRecord | None = None
    case_id: str | None = None
    batch_ordinal: int | None = None

    def __post_init__(self) -> None:
        if (
            self.manifest_schema_name != RUN_MANIFEST_SCHEMA_NAME
            or type(self.manifest_schema_version) is not int
            or self.manifest_schema_version != RUN_MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError("unsupported local run manifest schema")
        validate_run_id(self.run_id)
        if not isinstance(self.status, RunStatus) or not isinstance(self.mode, RunMode):
            raise ValueError("invalid run status or mode")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed is outside the RNG contract")
        RunPlan(self.total_steps, self.chunk_steps)
        validate_sha256(self.config_hash)
        if self.compile_signature is not None:
            validate_sha256(self.compile_signature)
        if type(self.initial_step) is not int or self.initial_step < 0:
            raise ValueError("initial_step must be a nonnegative integer")
        if self.steps_executed is not None and (
            type(self.steps_executed) is not int or not 0 <= self.steps_executed <= self.total_steps
        ):
            raise ValueError("invalid steps_executed")
        created = _timestamp(self.created_at_utc)
        started = None if self.started_at_utc is None else _timestamp(self.started_at_utc)
        finished = None if self.finished_at_utc is None else _timestamp(self.finished_at_utc)
        if (started is not None and started < created) or (
            finished is not None and finished < (started or created)
        ):
            raise ValueError("lifecycle timestamps are out of order")
        if self.status in (RunStatus.CREATED, RunStatus.RUNNING):
            if (
                finished is not None
                or self.final_step is not None
                or self.error is not None
                or self.steps_executed != 0
            ):
                raise ValueError("nonterminal manifest contains terminal observations")
            if (self.status == RunStatus.CREATED) != (started is None):
                raise ValueError("started timestamp does not match status")
        elif self.status == RunStatus.COMPLETED:
            if (
                started is None
                or finished is None
                or self.error is not None
                or self.steps_executed != self.total_steps
                or type(self.final_step) is not int
                or self.final_step != self.initial_step + self.total_steps
            ):
                raise ValueError("inconsistent completed manifest")
        elif finished is None or self.error is None:
            raise ValueError("failed run requires error and finish timestamp")
        if self.config_artifact != "config.json":
            raise ValueError("config_artifact must be config.json")
        paths = [validate_relative_path(a.relative_path) for a in self.artifacts]
        names = [a.name for a in self.artifacts]
        if len(set(paths)) != len(paths) or len(set(names)) != len(names):
            raise ValueError("artifact paths and names must be unique")
        if "config.json" not in paths:
            raise ValueError("configuration artifact is required")
        if self.status == RunStatus.COMPLETED and "summary.json" not in paths:
            raise ValueError("completed run requires a summary artifact")
        if self.case_id is not None and not self.case_id.strip():
            raise ValueError("case_id must be nonempty")
        if self.batch_ordinal is not None and (
            type(self.batch_ordinal) is not int or self.batch_ordinal < 1
        ):
            raise ValueError("batch ordinal must be positive")


def manifest_to_payload(manifest: RunManifest) -> dict[str, object]:
    return cast(dict[str, object], asdict(manifest))


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _invalid_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


def load_run_manifest(path: Path) -> RunManifest:
    """Validate JSON types, paths, identities, schema, and lifecycle consistency."""
    try:
        text = path.read_text(encoding="utf-8")
        json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_invalid_constant)
        return TypeAdapter(RunManifest).validate_json(text, strict=True)
    except (ValueError, ValidationError, UnicodeError) as exc:
        raise ManifestValidationError(f"invalid manifest {path}: {exc}") from exc
