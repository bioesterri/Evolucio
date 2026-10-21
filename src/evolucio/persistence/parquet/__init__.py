"""Host-only incremental Parquet telemetry and indexed lazy readers."""

from .reader import scan_births, scan_deaths, scan_genealogy, scan_metrics
from .schemas import ParquetPersistencePolicy
from .verify import ParquetDatasetVerification, verify_parquet_datasets
from .writer import ParquetDatasetSummary, ParquetRunConsumer

__all__ = [
    "ParquetDatasetSummary",
    "ParquetDatasetVerification",
    "ParquetPersistencePolicy",
    "ParquetRunConsumer",
    "scan_births",
    "scan_deaths",
    "scan_genealogy",
    "scan_metrics",
    "verify_parquet_datasets",
]
