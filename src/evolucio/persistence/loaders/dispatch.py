"""Version dispatch to the existing format-specific parsers."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

import polars as pl

from evolucio.config import load_config
from evolucio.config.versions import CONFIG_SCHEMA_VERSION
from evolucio.persistence.genomes import read_genome_index
from evolucio.persistence.local.manifest import load_run_manifest
from evolucio.persistence.local.schema import RUN_MANIFEST_SCHEMA_NAME
from evolucio.persistence.parquet.index import load_index as load_parquet
from evolucio.persistence.parquet.reader import scan_dataset
from evolucio.persistence.zarr.index import load_index as load_zarr
from evolucio.persistence.zarr.snapshot import load_snapshot

from .errors import DataSchemaError, UnsupportedSchemaVersionError
from .registry import CONFIG, GENOMES, PARQUET, SNAPSHOT, TRACKING, ZARR, SchemaDescriptor, Version
from .tracking import load_tracking_state

type Parser = Callable[[Path], object]


def _scan(path: Path, *, name: str) -> pl.LazyFrame:
    return scan_dataset(path, name)


# Each parser applies the structural contract for its format. The run audit
# adds physical verification and cross-artifact checks independently.
_PARSERS: dict[tuple[str, Version], Parser] = {
    ("local_run_manifest_v1", 1): load_run_manifest,
    (RUN_MANIFEST_SCHEMA_NAME, 2): load_run_manifest,
    (CONFIG, CONFIG_SCHEMA_VERSION): load_config,
    (PARQUET, 1): load_parquet,
    (PARQUET, 2): load_parquet,
    (ZARR, 1): load_zarr,
    (ZARR, 2): load_zarr,
    (GENOMES, 1): read_genome_index,
    (SNAPSHOT, 1): load_snapshot,
    (TRACKING, 1): load_tracking_state,
    **{
        (name, 1): partial(_scan, name=name)
        for name in ("metrics", "births", "deaths", "genealogy")
    },
}


def load_versioned[T](path: Path, schema: SchemaDescriptor, result_type: type[T]) -> T:
    parser = _PARSERS.get((schema.name, schema.version))
    if parser is None:
        supported = tuple(version for name, version in _PARSERS if name == schema.name)
        raise UnsupportedSchemaVersionError(schema.name, schema.version, supported)
    result = parser(path)
    if not isinstance(result, result_type):
        raise DataSchemaError(f"Invalid {schema.name} document at {path}")
    return result
