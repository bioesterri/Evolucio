"""Errors raised while identifying and loading persisted data."""


class DataSchemaError(ValueError):
    """Persisted metadata or data does not match its declared schema."""


class UnsupportedSchemaError(DataSchemaError):
    """The declared schema is unknown."""


class UnsupportedSchemaVersionError(UnsupportedSchemaError):
    def __init__(
        self, schema_name: str, found_version: int | str, supported_versions: tuple[int | str, ...]
    ) -> None:
        self.schema_name = schema_name
        self.found_version = found_version
        self.supported_versions = supported_versions
        super().__init__(
            f"{schema_name} version {found_version!r}; supported versions: {supported_versions}"
        )


class MissingArtifactError(FileNotFoundError):
    """A required or requested artifact is absent."""


class ArtifactIntegrityError(ValueError):
    """Physical integrity could not be established."""


class DataConsistencyError(ValueError):
    """Persisted records disagree."""


class RunDataCompatibilityError(ValueError):
    """A run cannot be used for the requested operation."""
