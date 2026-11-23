"""Content-addressed storage of selected neural policies."""

import hashlib
import json
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast
from zipfile import BadZipFile

import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evolucio.core.policy.genome_schema import GENOME_SCHEMA_NAME, GENOME_SCHEMA_VERSION
from evolucio.core.policy.model import PolicyMLP, policy_from_parameters
from evolucio.core.policy.schema import (
    POLICY_PARAMETER_COUNT,
    POLICY_PARAMETER_SPECS,
    POLICY_SCHEMA_NAME,
    POLICY_SCHEMA_VERSION,
)
from evolucio.persistence.local.artifacts import artifact_path, sha256_file, validate_sha256

HIGHLIGHTED_GENOME_STORE_SCHEMA_VERSION = 1
INDEX_PATH = "artifacts/genomes/index.json"
_ROOT = "artifacts/genomes"
_KEYS = tuple(spec.path.replace(".", "_") for spec in POLICY_PARAMETER_SPECS)
type GenomeLeaves = tuple[np.ndarray[Any, np.dtype[np.float32]], ...]


class GenomeHighlightReason(StrEnum):
    FOUNDER = "founder"
    FINAL_LINEAGE_REPRESENTATIVE = "final_lineage_representative"


class HighlightedGenomeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    genome_id: int = Field(ge=0)
    genome_hash: str
    agent_id: int = Field(ge=0)
    parent_genome_id: int | None = Field(ge=0)
    lineage_id: int = Field(ge=0)
    generation: int = Field(ge=0)
    reasons: dict[GenomeHighlightReason, int]
    relative_path: str
    artifact_sha256: str
    byte_size: int = Field(gt=0)
    parameter_count: int = POLICY_PARAMETER_COUNT
    policy_schema_name: str = POLICY_SCHEMA_NAME
    policy_schema_version: int = POLICY_SCHEMA_VERSION

    @model_validator(mode="after")
    def check_metadata(self) -> "HighlightedGenomeRecord":
        validate_sha256(self.genome_hash)
        validate_sha256(self.artifact_sha256)
        if self.relative_path != f"{_ROOT}/{self.genome_hash}.npz":
            raise ValueError("invalid genome content path")
        if (self.policy_schema_name, self.policy_schema_version, self.parameter_count) != (
            POLICY_SCHEMA_NAME,
            POLICY_SCHEMA_VERSION,
            POLICY_PARAMETER_COUNT,
        ):
            raise ValueError("incompatible policy schema")
        if not self.reasons or min(self.reasons.values()) < 0:
            raise ValueError("highlight reasons require nonnegative steps")
        if (self.generation == 0) != (self.parent_genome_id is None):
            raise ValueError("inconsistent genome ancestry")
        if GenomeHighlightReason.FOUNDER in self.reasons and self.generation != 0:
            raise ValueError("founder highlight requires generation zero")
        return self


class GenomeIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: int = HIGHLIGHTED_GENOME_STORE_SCHEMA_VERSION
    run_id: str
    genomes: tuple[HighlightedGenomeRecord, ...]

    @model_validator(mode="after")
    def check_index(self) -> "GenomeIndex":
        if self.schema_version != HIGHLIGHTED_GENOME_STORE_SCHEMA_VERSION or not self.run_id:
            raise ValueError("unsupported genome index version or missing run identity")
        if len({record.genome_id for record in self.genomes}) != len(self.genomes):
            raise ValueError("duplicate genome ID")
        if len({record.agent_id for record in self.genomes}) != len(self.genomes):
            raise ValueError("duplicate highlighted agent")
        lineages = [
            record.lineage_id
            for record in self.genomes
            if GenomeHighlightReason.FINAL_LINEAGE_REPRESENTATIVE in record.reasons
        ]
        if len(set(lineages)) != len(lineages):
            raise ValueError("multiple final representatives for one lineage")
        return self


def genome_content_hash(leaves: GenomeLeaves) -> str:
    """Hash a canonical little-endian float32 policy in schema leaf order."""
    digest = hashlib.sha256(f"{GENOME_SCHEMA_NAME}:{GENOME_SCHEMA_VERSION}\n".encode())
    for spec, leaf in zip(POLICY_PARAMETER_SPECS, leaves, strict=True):
        if leaf.dtype != np.dtype("float32") or leaf.shape != spec.shape:
            raise ValueError(f"invalid dtype or shape for {spec.path}")
        if not leaf.flags.c_contiguous or not np.isfinite(leaf).all():
            raise ValueError(f"expected finite C-order tensor for {spec.path}")
        header = json.dumps([spec.path, "<f4", spec.shape], separators=(",", ":"))
        digest.update(header.encode() + b"\n")
        digest.update(leaf.astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def read_genome_index(run_dir: Path) -> GenomeIndex:
    index = GenomeIndex.model_validate_json(artifact_path(run_dir, INDEX_PATH).read_bytes())
    if "schema_version" not in index.model_fields_set or any(
        not {"parameter_count", "policy_schema_name", "policy_schema_version"}
        <= record.model_fields_set
        for record in index.genomes
    ):
        raise ValueError("missing genome index or policy schema metadata")
    return index


def _read_leaves(path: Path) -> GenomeLeaves:
    with np.load(path, allow_pickle=False) as archive:
        if sorted(archive.files) != sorted(_KEYS):
            raise ValueError(f"invalid tensor keys in {path.name}")
        return cast("GenomeLeaves", tuple(archive[key] for key in _KEYS))


def store_genome(run_dir: Path, leaves: GenomeLeaves) -> tuple[str, str, str, int]:
    digest = genome_content_hash(leaves)
    relative = f"{_ROOT}/{digest}.npz"
    path = artifact_path(run_dir, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with NamedTemporaryFile(dir=path.parent, suffix=".npz") as temporary:
            w1, b1, w2, b2 = leaves
            np.savez(
                temporary.file,
                layer1_weight=w1,
                layer1_bias=b1,
                layer2_weight=w2,
                layer2_bias=b2,
            )
            temporary.file.flush()
            with suppress(FileExistsError):
                path.hardlink_to(temporary.name)
    if genome_content_hash(_read_leaves(path)) != digest:
        raise ValueError(f"semantic genome hash mismatch for {relative}")
    return digest, relative, sha256_file(path), path.stat().st_size


def read_genome_leaves(run_dir: Path, record: HighlightedGenomeRecord) -> GenomeLeaves:
    path = artifact_path(run_dir, record.relative_path)
    if sha256_file(path) != record.artifact_sha256 or path.stat().st_size != record.byte_size:
        raise ValueError(f"file SHA or size mismatch for {record.relative_path}")
    leaves = _read_leaves(path)
    if genome_content_hash(leaves) != record.genome_hash:
        raise ValueError(f"semantic genome hash mismatch for {record.relative_path}")
    return leaves


@dataclass(frozen=True, slots=True)
class GenomeStoreVerification:
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def verify_genome_store(run_dir: Path) -> GenomeStoreVerification:
    errors: list[str] = []
    try:
        index = read_genome_index(run_dir)
        referenced: dict[str, tuple[str, int]] = {}
        for record in index.genomes:
            identity = (record.artifact_sha256, record.byte_size)
            if record.relative_path in referenced:
                if referenced[record.relative_path] != identity:
                    errors.append(f"incompatible references for {record.relative_path}")
                continue
            referenced[record.relative_path] = identity
            try:
                read_genome_leaves(run_dir, record)
            except (OSError, ValueError, BadZipFile, EOFError) as exc:
                errors.append(str(exc))
        for path in sorted(artifact_path(run_dir, _ROOT).glob("*.npz")):
            if f"{_ROOT}/{path.name}" not in referenced:
                errors.append(f"orphan genome file: {path.name}")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    return GenomeStoreVerification(tuple(errors))


def load_highlighted_genome(run_dir: Path, genome_id: int) -> PolicyMLP:
    index = read_genome_index(run_dir)
    for record in index.genomes:
        if record.genome_id == genome_id:
            w1, b1, w2, b2 = (
                jnp.asarray(value)  # pyright: ignore[reportUnknownMemberType]
                for value in read_genome_leaves(run_dir, record)
            )
            return policy_from_parameters(
                layer1_weight=w1,
                layer1_bias=b1,
                layer2_weight=w2,
                layer2_bias=b2,
            )
    raise KeyError(genome_id)
