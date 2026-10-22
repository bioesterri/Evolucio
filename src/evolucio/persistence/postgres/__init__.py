"""Optional PostgreSQL relational registry (requires the persistence extra)."""

from .errors import (
    ArtifactIndexConflictError,
    DatabaseSchemaMismatchError,
    DatabaseUnavailableError,
    InvalidRunTransitionError,
    RegistryIntegrityError,
    RunAlreadyExistsError,
)
from .reconcile import reconcile_run, sync_run_artifacts
from .registry import (
    DatabaseStatus,
    check_database_connection,
    create_registry_engine,
    create_session_factory,
    database_status,
    upgrade_database,
)
from .repository import ExperimentRepository, PostgresRegistry, RunRepository
from .types import ComparisonRun, ExperimentRecord, ReconciliationReport, RunSummaryRecord

__all__ = [
    "ArtifactIndexConflictError",
    "ComparisonRun",
    "DatabaseSchemaMismatchError",
    "DatabaseStatus",
    "DatabaseUnavailableError",
    "ExperimentRecord",
    "ExperimentRepository",
    "InvalidRunTransitionError",
    "PostgresRegistry",
    "ReconciliationReport",
    "RegistryIntegrityError",
    "RunAlreadyExistsError",
    "RunRepository",
    "RunSummaryRecord",
    "check_database_connection",
    "create_registry_engine",
    "create_session_factory",
    "database_status",
    "reconcile_run",
    "sync_run_artifacts",
    "upgrade_database",
]
