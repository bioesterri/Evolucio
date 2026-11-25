"""Explicit analytical exclusion recorded outside the historical lifecycle."""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from pydantic import TypeAdapter

from .artifacts import artifact_path, atomic_write_text, json_text
from .manifest import RunStatus, load_run_manifest, utc_now, validate_run_id
from .ownership import run_ownership

EXCLUSION_PATH = "artifacts/analysis/exclusion.json"


@dataclass(frozen=True, slots=True)
class AnalysisExclusion:
    run_id: str
    reason: str
    discarded_at_utc: str

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        if not 0 < len(self.reason.strip()) <= 4096:
            raise ValueError("exclusion requires a nonempty reason of at most 4096 characters")
        if not self.discarded_at_utc.endswith("Z"):
            raise ValueError("exclusion time must be UTC")
        datetime.fromisoformat(self.discarded_at_utc)


def read_exclusion(run_dir: Path, run_id: str) -> AnalysisExclusion | None:
    path = artifact_path(run_dir, EXCLUSION_PATH)
    if not path.is_file():
        return None
    record = TypeAdapter(AnalysisExclusion).validate_json(path.read_bytes(), strict=True)
    if record.run_id != run_id:
        raise ValueError("analytical exclusion belongs to a different run")
    return record


def discard_from_analysis(run_dir: Path, reason: str) -> AnalysisExclusion:
    with run_ownership(run_dir) as ownership:
        if ownership != "inactive":
            raise ValueError(f"exclusive run ownership unavailable: {ownership}")
        manifest = load_run_manifest(run_dir / "manifest.json")
        if manifest.status not in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.STOPPED):
            raise ValueError("only a terminal run can be excluded from analysis")
        existing = read_exclusion(run_dir, manifest.run_id)
        if existing is not None:
            if reason.strip() != existing.reason:
                raise ValueError("analytical exclusion already has a different reason")
            return existing
        record = AnalysisExclusion(manifest.run_id, reason.strip(), utc_now())
        path = artifact_path(run_dir, EXCLUSION_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json_text(asdict(record)))
        return record
