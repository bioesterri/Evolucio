"""Read-only verification of a transportable local run directory."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from evolucio.config import freeze_config, load_config
from evolucio.config.errors import ConfigError

from .artifacts import InvalidArtifactPath, artifact_path, sha256_file
from .manifest import load_run_manifest


@dataclass(frozen=True, slots=True)
class LocalRunVerification:
    ok: bool
    missing_artifacts: tuple[str, ...]
    hash_mismatches: tuple[str, ...]
    invalid_paths: tuple[str, ...]


def verify_local_run(run_dir: Path) -> LocalRunVerification:
    """Raise for invalid input/manifest; report artifact integrity without writing."""
    if (run_dir / "manifest.json").is_symlink():
        raise InvalidArtifactPath("manifest.json cannot be a symlink")
    manifest = load_run_manifest(run_dir / "manifest.json")
    missing: list[str] = []
    mismatches: list[str] = []
    invalid: list[str] = []
    if run_dir.name != manifest.run_id:
        invalid.append("run_id")
    for record in manifest.artifacts:
        relative = record.relative_path
        try:
            path = artifact_path(run_dir, relative)
        except InvalidArtifactPath:
            invalid.append(relative)
            continue
        if not path.is_file():
            missing.append(relative)
            continue
        if path.stat().st_size != record.byte_size or sha256_file(path) != record.sha256:
            mismatches.append(relative)
            continue
        if relative == manifest.config_artifact:
            try:
                config = load_config(path)
                if (
                    freeze_config(config).config_hash != manifest.config_hash
                    or config.seed != manifest.seed
                ):
                    mismatches.append(relative)
            except (ConfigError, ValidationError):
                mismatches.append(relative)
    return LocalRunVerification(
        not (missing or mismatches or invalid), tuple(missing), tuple(mismatches), tuple(invalid)
    )
