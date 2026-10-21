"""Local recording lifecycle, independent of simulation execution."""

from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

from evolucio.config import ExperimentConfig, freeze_config, serialize_config
from evolucio.config.versions import CONFIG_SCHEMA_VERSION
from evolucio.host.schema import HOST_DTO_SCHEMA_DIGEST
from evolucio.runtime.modes import RunMode
from evolucio.runtime.types import RunPlan

from . import version_info
from .artifacts import ArtifactRecord, LocalArtifactRegistry, atomic_write_text, json_text
from .manifest import (
    RunErrorRecord,
    RunManifest,
    RunSchemas,
    RunStatus,
    manifest_to_payload,
    utc_now,
    validate_run_id,
)
from .schema import RUN_MANIFEST_SCHEMA_NAME, RUN_MANIFEST_SCHEMA_VERSION

DEFAULT_RUNS_ROOT = Path("runs")


class LocalRunRecorder:
    """Write metadata transitions only; never initialize or execute a simulation."""

    def __init__(
        self,
        run_dir: Path,
        manifest: RunManifest,
        registry: LocalArtifactRegistry,
    ) -> None:
        self.run_dir = run_dir
        self._manifest = manifest
        self._registry = registry

    @property
    def run_id(self) -> str:
        return self._manifest.run_id

    @property
    def manifest(self) -> RunManifest:
        return self._manifest

    def _save(self, manifest: RunManifest) -> None:
        atomic_write_text(self.run_dir / "manifest.json", json_text(manifest_to_payload(manifest)))
        self._manifest = manifest

    def _require_open(self) -> None:
        if self.manifest.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            raise ValueError("terminal runs cannot be changed or reopened")

    def mark_running(self) -> None:
        if self.manifest.status != RunStatus.CREATED:
            raise ValueError("only a created run can start")
        self._save(replace(self.manifest, status=RunStatus.RUNNING, started_at_utc=utc_now()))

    def mark_completed(
        self,
        *,
        summary: dict[str, object],
        steps_executed: int,
        final_step: int,
    ) -> None:
        if self.manifest.status != RunStatus.RUNNING:
            raise ValueError("only a running run can complete")
        if (
            type(steps_executed) is not int
            or steps_executed != self.manifest.total_steps
            or type(final_step) is not int
            or final_step != self.manifest.initial_step + steps_executed
        ):
            raise ValueError("completion must cover the entire run plan")
        payload = summary | {
            "run_id": self.run_id,
            "seed": self.manifest.seed,
            "status": RunStatus.COMPLETED.value,
            "steps_executed": steps_executed,
            "final_step": final_step,
        }
        self._registry.write_json_artifact(
            relative_path="summary.json", payload=payload, name="summary", kind="summary"
        )
        self._save(
            replace(
                self.manifest,
                status=RunStatus.COMPLETED,
                finished_at_utc=utc_now(),
                steps_executed=steps_executed,
                final_step=final_step,
                artifacts=self._registry.records,
            )
        )

    def mark_failed(self, error: RunErrorRecord) -> None:
        self._require_open()
        self._registry.write_json_artifact(
            relative_path="error.json", payload=asdict(error), name="error", kind="error"
        )
        self._save(
            replace(
                self.manifest,
                status=RunStatus.FAILED,
                finished_at_utc=utc_now(),
                steps_executed=None,
                final_step=None,
                error=error,
                artifacts=self._registry.records,
            )
        )

    def register_artifact(
        self,
        *,
        name: str,
        kind: str,
        relative_path: str,
        media_type: str,
    ) -> ArtifactRecord:
        self._require_open()
        record = self._registry.register_existing_artifact(
            name=name,
            kind=kind,
            relative_path=relative_path,
            media_type=media_type,
        )
        self._save(replace(self.manifest, artifacts=self._registry.records))
        return record

    def write_json_artifact(
        self,
        *,
        relative_path: str,
        payload: object,
        name: str,
        kind: str,
    ) -> ArtifactRecord:
        self._require_open()
        record = self._registry.write_json_artifact(
            relative_path=relative_path,
            payload=payload,
            name=name,
            kind=kind,
        )
        self._save(replace(self.manifest, artifacts=self._registry.records))
        return record


def create_local_run(
    *,
    config: ExperimentConfig,
    plan: RunPlan,
    mode: RunMode,
    output_root: Path = DEFAULT_RUNS_ROOT,
    run_id: str | None = None,
    compile_signature: str | None = None,
    schemas: RunSchemas | None = None,
    initial_step: int = 0,
    case_id: str | None = None,
    batch_ordinal: int | None = None,
) -> LocalRunRecorder:
    """Freeze the normalized host config and record identity before initialization."""
    identity = str(uuid4()) if run_id is None else validate_run_id(run_id)
    frozen = freeze_config(config)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / identity
    run_dir.mkdir(exist_ok=False)
    (run_dir / "artifacts").mkdir()
    registry = LocalArtifactRegistry(run_dir)
    atomic_write_text(run_dir / "config.json", serialize_config(config, format="json"))
    registry.register_existing_artifact(
        name="config",
        kind="configuration",
        relative_path="config.json",
        media_type="application/json",
    )
    manifest = RunManifest(
        manifest_schema_name=RUN_MANIFEST_SCHEMA_NAME,
        manifest_schema_version=RUN_MANIFEST_SCHEMA_VERSION,
        run_id=identity,
        status=RunStatus.CREATED,
        created_at_utc=utc_now(),
        started_at_utc=None,
        finished_at_utc=None,
        seed=config.seed,
        config_hash=frozen.config_hash,
        config_artifact="config.json",
        total_steps=plan.total_steps,
        chunk_steps=plan.chunk_steps,
        mode=mode,
        initial_step=initial_step,
        final_step=None,
        steps_executed=0,
        code_version=version_info.code_version_info(),
        environment=version_info.environment_info(),
        compile_signature=compile_signature,
        schemas=schemas or RunSchemas(CONFIG_SCHEMA_VERSION, HOST_DTO_SCHEMA_DIGEST),
        artifacts=registry.records,
        case_id=case_id,
        batch_ordinal=batch_ordinal,
    )
    recorder = LocalRunRecorder(run_dir, manifest, registry)
    atomic_write_text(run_dir / "manifest.json", json_text(manifest_to_payload(manifest)))
    return recorder
