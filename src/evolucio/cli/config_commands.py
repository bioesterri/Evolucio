"""Configuration loading and inspection through the canonical host API."""

from dataclasses import asdict
from pathlib import Path

from evolucio.config import FrozenConfig, freeze_config, load_config
from evolucio.config.errors import ConfigIOError

from .formatting import json_text, text_fields


def load_frozen(path: Path) -> FrozenConfig:
    """Read exactly the supplied file and preserve canonical validation."""
    if not path.exists():
        raise ConfigIOError(f"file does not exist: {path}")
    if not path.is_file():
        raise ConfigIOError(f"not a file: {path}")
    return freeze_config(load_config(path))


def identity(frozen: FrozenConfig) -> dict[str, object]:
    """Use the repository's existing scientific and compile identities."""
    from evolucio.config.compile import build_compile_signature, compile_signature_digest

    return {
        "schema_version": frozen.config.schema_version,
        "config_hash": frozen.config_hash,
        "compile_signature": compile_signature_digest(build_compile_signature(frozen.config)),
    }


def validate_config(path: Path) -> str:
    """Validate without compiling arrays, initializing state or running steps."""
    return "Configuration valid\n" + text_fields(identity(load_frozen(path)))


def inspect_config(path: Path, *, format: str) -> str:
    """Show normalized config and its existing static compile contract."""
    from evolucio.config.compile import build_compile_signature

    frozen = load_frozen(path)
    values = identity(frozen)
    values["configuration"] = frozen.config.model_dump(mode="json")
    values["static_compile_contract"] = asdict(build_compile_signature(frozen.config))
    return json_text(values) if format == "json" else text_fields(values)
