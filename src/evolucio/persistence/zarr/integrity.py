"""Read-only Zarr index, byte integrity, schema and compatibility verification."""

from dataclasses import dataclass
from pathlib import Path

from evolucio.persistence.local.artifacts import artifact_path

from .index import load_index
from .storage import CheckpointCompatibilityError, read_metadata, verify_store_tree


@dataclass(frozen=True, slots=True)
class ZarrArtifactVerification:
    ok: bool
    missing_stores: tuple[str, ...]
    hash_mismatches: tuple[str, ...]
    orphan_stores: tuple[str, ...]
    schema_errors: tuple[str, ...]
    compatibility_errors: tuple[str, ...]


def verify_zarr_artifacts(run_dir: Path) -> ZarrArtifactVerification:
    run_dir = run_dir.resolve()
    missing: list[str] = []
    hashes: list[str] = []
    schema_errors: list[str] = []
    compatibility: list[str] = []
    indexed: set[str] = set()
    orphans: list[str] = []
    try:
        index = load_index(run_dir)
        for record in (*index.checkpoints, *index.snapshots):
            relative = record.relative_path
            indexed.add(relative)
            path = artifact_path(run_dir, relative)
            if not path.is_dir():
                missing.append(relative)
                continue
            try:
                size, digest = verify_store_tree(path)
            except (OSError, ValueError):
                hashes.append(relative)
                continue
            if size != record.byte_size or digest != record.sha256:
                hashes.append(relative)
                continue
            try:
                metadata = read_metadata(path)
                if record.kind == "checkpoint":
                    from .checkpoint import read_checkpoint_arrays

                    read_checkpoint_arrays(path)
                else:
                    from .snapshot import read_snapshot

                    read_snapshot(path)
                if (
                    metadata["run_id"] != index.run_id
                    or metadata["state_step"] != record.state_step
                    or metadata["created_at_utc"] != record.created_at_utc
                ):
                    schema_errors.append(f"{relative}: index identity mismatch")
                if (
                    metadata["config_hash"] != index.config_hash
                    or metadata["compile_signature"] != index.compile_signature
                ):
                    compatibility.append(f"{relative}: incompatible index context")
            except CheckpointCompatibilityError as exc:
                compatibility.append(f"{relative}: {exc}")
            except (OSError, ValueError, KeyError) as exc:
                schema_errors.append(f"{relative}: {exc}")
    except (OSError, ValueError) as exc:
        schema_errors.append(str(exc))
    try:
        root = artifact_path(run_dir, "artifacts/zarr")
        for directory in (root / "checkpoints", root / "snapshots"):
            if directory.is_symlink():
                raise ValueError("symlink in Zarr capture directory")
            if directory.exists():
                orphans.extend(
                    child.relative_to(run_dir).as_posix()
                    for child in directory.iterdir()
                    if child.name.endswith(".zarr")
                    and child.relative_to(run_dir).as_posix() not in indexed
                )
    except (OSError, ValueError) as exc:
        schema_errors.append(str(exc))
    return ZarrArtifactVerification(
        not (missing or hashes or orphans or schema_errors or compatibility),
        tuple(missing),
        tuple(hashes),
        tuple(sorted(orphans)),
        tuple(schema_errors),
        tuple(compatibility),
    )
