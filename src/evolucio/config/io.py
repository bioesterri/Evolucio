"""Safe YAML and JSON configuration I/O."""

import json
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import ValidationError

from .errors import (
    ConfigFormatError,
    ConfigIOError,
    DuplicateConfigKeyError,
    UnsupportedConfigVersionError,
)
from .models import ExperimentConfig
from .versions import CONFIG_SCHEMA_VERSION

type ConfigFormat = Literal["yaml", "json"]


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateConfigKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = cast(
            object,
            loader.construct_object(key_node, deep=deep),  # pyright: ignore[reportUnknownMemberType]
        )
        if not isinstance(key, Hashable):
            raise ConfigFormatError("YAML mapping keys must be hashable")
        if key in result:
            raise DuplicateConfigKeyError(f"duplicate YAML key: {key!r}")
        result[key] = cast(
            object,
            loader.construct_object(value_node, deep=deep),  # pyright: ignore[reportUnknownMemberType]
        )
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def parse_config(text: str, *, format: ConfigFormat) -> ExperimentConfig:
    """Parse and validate one YAML or JSON document."""
    if not text.strip():
        raise ConfigFormatError(f"empty {format} configuration")
    try:
        value = (
            json.loads(text, object_pairs_hook=_unique_pairs)
            if format == "json"
            else yaml.load(text, Loader=_UniqueSafeLoader)
        )
    except DuplicateConfigKeyError:
        raise
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigFormatError(f"invalid {format} syntax: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ConfigFormatError(f"{format} document root must be an object")
    data = cast(dict[str, object], value)
    version = data.get("schema_version")
    if version is not None and version != CONFIG_SCHEMA_VERSION:
        raise UnsupportedConfigVersionError(
            f"unsupported schema_version {version!r}; expected {CONFIG_SCHEMA_VERSION!r}"
        )
    return ExperimentConfig.model_validate(data)


def load_config(path: str | Path) -> ExperimentConfig:
    """Read and validate a UTF-8 configuration file by extension."""
    source = Path(path)
    formats: dict[str, ConfigFormat] = {".yaml": "yaml", ".yml": "yaml", ".json": "json"}
    if source.suffix.lower() not in formats:
        raise ConfigFormatError(f"unsupported configuration extension: {source.suffix or '<none>'}")
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigIOError(f"cannot read configuration {source}: {exc}") from exc
    try:
        return parse_config(text, format=formats[source.suffix.lower()])
    except (ConfigFormatError, UnsupportedConfigVersionError, ValidationError) as exc:
        exc.add_note(f"configuration file: {source}")
        raise


def serialize_config(config: ExperimentConfig, *, format: ConfigFormat, pretty: bool = True) -> str:
    """Serialize a validated configuration, including defaults and nulls."""
    data = config.model_dump(mode="json", exclude_none=False, exclude_defaults=False)
    if format == "json":
        return (
            json.dumps(
                data,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2 if pretty else None,
                separators=None if pretty else (",", ":"),
            )
            + "\n"
        )
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=True, default_flow_style=not pretty)


def dump_config(config: ExperimentConfig, path: str | Path, *, overwrite: bool = False) -> None:
    """Write configuration without creating parents or overwriting by default."""
    target = Path(path)
    formats: dict[str, ConfigFormat] = {".yaml": "yaml", ".yml": "yaml", ".json": "json"}
    if target.suffix.lower() not in formats:
        raise ConfigFormatError(f"unsupported configuration extension: {target.suffix or '<none>'}")
    try:
        with target.open("w" if overwrite else "x", encoding="utf-8") as stream:
            stream.write(serialize_config(config, format=formats[target.suffix.lower()]))
    except OSError as exc:
        raise ConfigIOError(f"cannot write configuration {target}: {exc}") from exc
