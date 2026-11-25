"""Explicit schema dispatch. Compound configuration versions retain their wire type."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from evolucio.config.versions import CONFIG_SCHEMA_VERSION

from .errors import DataSchemaError, UnsupportedSchemaVersionError

type Version = int | str


@dataclass(frozen=True, slots=True)
class SchemaDescriptor:
    name: str
    version: Version
    digest: str | None = None


def read_object(path: Path) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise DataSchemaError(f"duplicate key {key!r} in {path}")
            result[key] = value
        return result

    def invalid(value: str) -> object:
        raise DataSchemaError(f"invalid JSON constant {value} in {path}")

    if not path.is_file():
        from .errors import MissingArtifactError

        raise MissingArtifactError(str(path))
    value: object = json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise DataSchemaError(f"expected JSON object in {path}")
    return cast(dict[str, object], value)


CONFIG = "config"
PARQUET = "run_telemetry_parquet_v1"
ZARR = "zarr_artifact_index"
GENOMES = "highlighted_genome_index"
RECOVERY = "recovery_provenance"
TRACKING = "mlflow_tracking_state"
TRACKING_PATH = "artifacts/mlflow/tracking.json"
CHECKPOINT = "simulation_checkpoint_zarr_v1"
SNAPSHOT = "analysis_snapshot_zarr_v1"

# Formats without an embedded name use the artifact kind or file role.
_VERSIONS: dict[str, tuple[Version, ...]] = {
    CONFIG: (CONFIG_SCHEMA_VERSION,),
    **dict.fromkeys(("config.observations", "config.policy", "config.genome"), (1,)),
    "local_run_manifest_v1": (1,),
    "local_run_manifest_v2": (2,),
    PARQUET: (1, 2),
    **dict.fromkeys(("metrics", "births", "deaths", "genealogy"), (1,)),
    ZARR: (1, 2),
    CHECKPOINT: (1,),
    SNAPSHOT: (1,),
    GENOMES: (1,),
    "founders_v1": (1,),
    "agent_lifecycle_summary_v1": (1,),
    "lineage_outcome_summary_v1": (1,),
    RECOVERY: (1,),
    TRACKING: (1,),
}


def get_supported_schemas() -> tuple[SchemaDescriptor, ...]:
    return tuple(
        SchemaDescriptor(name, version)
        for name, versions in _VERSIONS.items()
        for version in versions
    )


def is_schema_supported(name: str, version: Version) -> bool:
    return type(version) in (int, str) and version in _VERSIONS.get(name, ())


def get_schema_descriptor(name: str, version: Version) -> SchemaDescriptor:
    if not is_schema_supported(name, version):
        raise UnsupportedSchemaVersionError(name, version, _VERSIONS.get(name, ()))
    return SchemaDescriptor(name, version)


def identify(
    payload: dict[str, object], name: str, *, version_key: str = "schema_version"
) -> SchemaDescriptor:
    version = payload.get(version_key)
    if type(version) not in (int, str):
        raise DataSchemaError(f"{name} requires an explicit {version_key}")
    return get_schema_descriptor(name, cast(Version, version))
