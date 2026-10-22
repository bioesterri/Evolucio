"""Public, contextual errors for the relational registry."""


class PostgresRegistryError(RuntimeError):
    """Base error for PostgreSQL registry operations."""


class DatabaseUnavailableError(PostgresRegistryError):
    """The configured PostgreSQL database cannot be reached."""


class RegistryIntegrityError(PostgresRegistryError):
    """Relational and local registry metadata disagree."""


class RunAlreadyExistsError(RegistryIntegrityError):
    """A run with the requested PR-35 identity already exists."""


class InvalidRunTransitionError(RegistryIntegrityError):
    """A requested run lifecycle transition is invalid."""


class ArtifactIndexConflictError(RegistryIntegrityError):
    """An indexed path has conflicting immutable metadata."""


class DatabaseSchemaMismatchError(PostgresRegistryError):
    """The installed Alembic revision is not the application head."""
