"""Configuration error hierarchy."""


class ConfigError(Exception):
    """Base error for configuration operations."""


class ConfigIOError(ConfigError):
    """A configuration file could not be read or written."""


class ConfigFormatError(ConfigError):
    """Configuration syntax or document shape is invalid."""


class DuplicateConfigKeyError(ConfigFormatError):
    """A mapping contains a duplicate key."""


class UnsupportedConfigVersionError(ConfigError):
    """The document declares an unsupported schema version."""
