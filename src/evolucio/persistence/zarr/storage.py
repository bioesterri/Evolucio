"""Atomic local Zarr v3 stores and portable streaming tree identities."""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import zarr

from evolucio.host.dto import HostArray
from evolucio.persistence.local.artifacts import (
    artifact_path,
    atomic_write_text,
    json_text,
    sha256_file,
    validate_relative_path,
)

from .schemas import StrictModel

CATALOG = "integrity.json"


class CheckpointIntegrityError(ValueError):
    """A capture is incomplete, corrupt or structurally invalid; never repaired."""


class CheckpointCompatibilityError(ValueError):
    """The explicit schema, configuration or compilation contract is incompatible."""


class StoreFile(StrictModel):
    relative_path: str
    byte_size: int
    sha256: str


class StoreCatalog(StrictModel):
    schema_version: int
    files: list[StoreFile]


def _files(path: Path) -> dict[str, Path]:
    if path.is_symlink() or not path.is_dir():
        raise CheckpointIntegrityError(f"not a regular capture directory: {path}")
    files: dict[str, Path] = {}
    for child in path.rglob("*"):
        if child.is_symlink():
            raise CheckpointIntegrityError(f"symlink in capture: {child}")
        if child.is_file():
            files[child.relative_to(path).as_posix()] = child
    return files


def seal_store(path: Path) -> None:
    files = _files(path)
    records = [
        StoreFile(relative_path=name, byte_size=entry.stat().st_size, sha256=sha256_file(entry))
        for name, entry in sorted(files.items())
        if name != CATALOG
    ]
    atomic_write_text(
        path / CATALOG, json_text(StoreCatalog(schema_version=1, files=records).model_dump())
    )


def verify_store_tree(path: Path) -> tuple[int, str]:
    """Hash catalogued files and catalog; reject missing, modified or extra Zarr files."""
    try:
        actual = _files(path)
        catalog = StoreCatalog.model_validate_json((path / CATALOG).read_text())
        if catalog.schema_version != 1:
            raise CheckpointIntegrityError("unsupported store integrity schema")
        expected = {CATALOG}
        for record in catalog.files:
            name = validate_relative_path(record.relative_path)
            if name in expected:
                raise CheckpointIntegrityError("duplicate or recursive store catalog entry")
            expected.add(name)
            file = artifact_path(path, name)
            if (
                not file.is_file()
                or file.stat().st_size != record.byte_size
                or sha256_file(file) != record.sha256
            ):
                raise CheckpointIntegrityError(f"missing or modified capture file: {name}")
        for name in actual.keys() - expected:
            if Path(name).name == "zarr.json" or "c" in Path(name).parts:
                raise CheckpointIntegrityError(f"uncatalogued Zarr file: {name}")
        digest = hashlib.sha256(b"evolucio-zarr-tree-v1\0")
        byte_size = 0
        for name in sorted(expected):
            file = actual[name]
            encoded = name.encode("utf-8")
            size = file.stat().st_size
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(size.to_bytes(8, "big"))
            with file.open("rb") as stream:
                while data := stream.read(1024 * 1024):
                    digest.update(data)
            byte_size += size
        return byte_size, digest.hexdigest()
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, CheckpointIntegrityError):
            raise
        raise CheckpointIntegrityError(f"invalid capture integrity: {exc}") from exc


def read_arrays(
    path: Path, specs: Mapping[str, tuple[tuple[int, ...], str]]
) -> tuple[dict[str, object], dict[str, HostArray]]:
    group = zarr.open_group(str(path), mode="r", use_consolidated=False)  # pyright: ignore[reportUnknownMemberType]
    arrays: dict[str, zarr.Array[Any]] = {}
    groups: set[str] = set()
    # Inventory explicit v3 metadata, excluding the integrity sidecar from Zarr traversal.
    for name, file in _files(path).items():
        if name == "zarr.json" or not name.endswith("/zarr.json"):
            continue
        node_path = name.removesuffix("/zarr.json")
        node_type = json.loads(file.read_text())["node_type"]
        if node_type == "array":
            arrays[node_path] = cast(zarr.Array[Any], group[node_path])
        elif node_type == "group":
            groups.add(node_path)
        else:
            raise CheckpointCompatibilityError(f"unsupported Zarr node: {name}")
    expected_groups = {name.rsplit("/", 1)[0] for name in specs if "/" in name}
    if set(arrays) != set(specs) or groups != expected_groups:
        raise CheckpointCompatibilityError("capture arrays/groups do not match the explicit schema")
    values: dict[str, HostArray] = {}
    for name, (shape, dtype) in specs.items():
        array = arrays[name]
        if array.shape != shape or np.dtype(array.dtype) != np.dtype(dtype):
            raise CheckpointCompatibilityError(f"incompatible shape/dtype: {name}")
        values[name] = np.asarray(array[...])
    return dict(group.attrs), values


def read_metadata(path: Path) -> dict[str, object]:
    return dict(zarr.open_group(str(path), mode="r", use_consolidated=False).attrs)  # pyright: ignore[reportUnknownMemberType]


def write_store(
    path: Path,
    arrays: Mapping[str, HostArray],
    metadata: dict[str, object],
    validate: Callable[[Path], object],
) -> tuple[int, str]:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable capture already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=path.parent, prefix=f"{path.name}.tmp-"))
    try:
        group = zarr.open_group(str(temporary), mode="w", zarr_format=3)  # pyright: ignore[reportUnknownMemberType]
        for name, array in arrays.items():
            chunks = tuple(max(1, min(size, 128)) for size in array.shape)
            group.create_array(name, data=array, chunks=chunks)
        group.attrs.update(cast(Any, metadata))
        seal_store(temporary)
        result = verify_store_tree(temporary)
        validate(temporary)
        if path.exists():
            raise FileExistsError(path)
        os.rename(temporary, path)
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
