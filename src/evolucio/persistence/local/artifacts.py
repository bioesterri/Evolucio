"""Safe local paths, atomic JSON, and immutable artifact byte identities."""

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


class InvalidArtifactPath(ValueError):
    """Artifact paths must stay within their owning run directory."""


def validate_relative_path(value: str) -> str:
    """Accept normalized portable POSIX paths, excluding the manifest itself."""
    if (
        not value
        or "\\" in value
        or "\0" in value
        or ":" in value
        or PureWindowsPath(value).drive
        or PurePosixPath(value).is_absolute()
        or any(part in (".", "..", "") for part in value.split("/"))
        or value == "manifest.json"
    ):
        raise InvalidArtifactPath(f"invalid artifact path: {value!r}")
    return value


def artifact_path(run_dir: Path, relative_path: str) -> Path:
    """Reject symlinks as well as lexical escapes; artifacts belong to one run."""
    validate_relative_path(relative_path)
    root = run_dir.resolve()
    path = root
    for part in relative_path.split("/"):
        path = path / part
        if path.is_symlink():
            raise InvalidArtifactPath(f"symlink in artifact path: {relative_path!r}")
    if not path.resolve().is_relative_to(root):
        raise InvalidArtifactPath(f"artifact escapes run: {relative_path!r}")
    return path


def validate_sha256(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("sha256 must contain 64 lowercase hexadecimal characters")


def sha256_file(path: Path) -> str:
    """Hash binary chunks with bounded memory, regardless of file size."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_text(payload: object) -> str:
    """Stable UTF-8 JSON text; serialization errors are never suppressed."""
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    """Flush a same-directory temporary file before replacing the target."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Exact bytes of one immutable artifact, relative to its run."""

    name: str
    kind: str
    relative_path: str
    media_type: str
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.name or not self.kind or not self.media_type:
            raise ValueError("artifact name, kind and media_type must be nonempty")
        validate_relative_path(self.relative_path)
        validate_sha256(self.sha256)
        if isinstance(self.byte_size, bool) or self.byte_size < 0:
            raise ValueError("artifact byte_size must be nonnegative")


class LocalArtifactRegistry:
    """One run's append-only artifact index; no device-library dependency."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._records: tuple[ArtifactRecord, ...] = ()

    @property
    def records(self) -> tuple[ArtifactRecord, ...]:
        return self._records

    def register_existing_artifact(
        self,
        *,
        name: str,
        kind: str,
        relative_path: str,
        media_type: str,
    ) -> ArtifactRecord:
        path = artifact_path(self.run_dir, relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"artifact is not a file: {relative_path}")
        record = ArtifactRecord(
            name, kind, relative_path, media_type, path.stat().st_size, sha256_file(path)
        )
        for existing in self._records:
            if existing.relative_path == relative_path or existing.name == name:
                if existing != record:
                    raise ValueError("artifact content or metadata conflicts with existing record")
                return existing
        self._records += (record,)
        return record

    def write_json_artifact(
        self,
        *,
        relative_path: str,
        payload: object,
        name: str,
        kind: str,
    ) -> ArtifactRecord:
        path = artifact_path(self.run_dir, relative_path)
        text = json_text(payload)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        candidate = ArtifactRecord(
            name, kind, relative_path, "application/json", len(text.encode("utf-8")), digest
        )
        for existing in self._records:
            if existing.relative_path == relative_path or existing.name == name:
                if existing != candidate:
                    raise ValueError("cannot overwrite an immutable artifact")
                return self.register_existing_artifact(
                    name=name, kind=kind, relative_path=relative_path, media_type="application/json"
                )
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, text)
        return self.register_existing_artifact(
            name=name, kind=kind, relative_path=relative_path, media_type="application/json"
        )
