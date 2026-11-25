"""Portable provenance for materialized continuations."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from evolucio.persistence.local.artifacts import artifact_path, sha256_file, validate_sha256
from evolucio.persistence.local.manifest import RunManifest, validate_run_id

PROVENANCE_PATH = "artifacts/recovery/provenance.json"


@dataclass(frozen=True, slots=True)
class RecoveryProvenance:
    schema_version: int
    source_run_id: str
    root_run_id: str
    recovery_depth: int
    source_checkpoint_step: int
    source_checkpoint_hash: str
    source_config_hash: str
    inherited_steps: int
    new_steps_requested: int
    target_step: int
    inherited_metrics_rows: int
    inherited_birth_rows: int
    inherited_death_rows: int
    inherited_genealogy_rows: int
    source_prefix_digest: str

    def __post_init__(self) -> None:
        validate_run_id(self.source_run_id)
        validate_run_id(self.root_run_id)
        for digest in (
            self.source_checkpoint_hash,
            self.source_config_hash,
            self.source_prefix_digest,
        ):
            validate_sha256(digest)
        if (
            self.schema_version != 1
            or self.recovery_depth < 1
            or (self.recovery_depth == 1) != (self.root_run_id == self.source_run_id)
            or self.inherited_steps != self.source_checkpoint_step
            or self.inherited_steps < 0
            or self.new_steps_requested <= 0
            or self.inherited_steps + self.new_steps_requested != self.target_step
            or min(
                self.inherited_metrics_rows,
                self.inherited_birth_rows,
                self.inherited_death_rows,
                self.inherited_genealogy_rows,
            )
            < 0
        ):
            raise ValueError("invalid recovery provenance")


def read_provenance(run_dir: Path, manifest: RunManifest) -> RecoveryProvenance | None:
    for record in manifest.artifacts:
        if record.relative_path == PROVENANCE_PATH:
            path = artifact_path(run_dir, PROVENANCE_PATH)
            if sha256_file(path) != record.sha256:
                raise ValueError("recovery provenance hash differs")
            provenance = TypeAdapter(RecoveryProvenance).validate_json(
                path.read_bytes(), strict=True
            )
            if (
                manifest.run_id in (provenance.source_run_id, provenance.root_run_id)
                or manifest.config_hash != provenance.source_config_hash
                or manifest.initial_step != provenance.source_checkpoint_step
                or manifest.total_steps != provenance.new_steps_requested
            ):
                raise ValueError("recovery provenance disagrees with the manifest")
            return provenance
    return None
