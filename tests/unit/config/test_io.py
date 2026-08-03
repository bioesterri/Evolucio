from pathlib import Path

import pytest
from pydantic import ValidationError

from evolucio.config import (
    ExperimentConfig,
    dump_config,
    load_config,
    parse_config,
    serialize_config,
)
from evolucio.config.errors import (
    ConfigFormatError,
    ConfigIOError,
    DuplicateConfigKeyError,
    UnsupportedConfigVersionError,
)


def test_load_formats_equivalent(config: ExperimentConfig, valid_path: Path) -> None:
    assert load_config(valid_path) == load_config(valid_path.with_suffix(".json")) == config


@pytest.mark.parametrize(
    ("text", "fmt", "error"),
    [
        ("", "yaml", ConfigFormatError),
        ("[]", "json", ConfigFormatError),
        ("{", "json", ConfigFormatError),
        ("x: [", "yaml", ConfigFormatError),
        ("a: 1\na: 2", "yaml", DuplicateConfigKeyError),
        ('{"a":1,"a":2}', "json", DuplicateConfigKeyError),
        ("!!python/object:os.system {}", "yaml", ConfigFormatError),
    ],
)
def test_bad_documents(text: str, fmt: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        parse_config(text, format=fmt)  # type: ignore[arg-type]


def test_public_apis_reject_unknown_format(config: ExperimentConfig) -> None:
    with pytest.raises(ConfigFormatError, match="toml"):
        parse_config("schema_version = '1.0'", format="toml")  # type: ignore[arg-type]
    with pytest.raises(ConfigFormatError, match="toml"):
        serialize_config(config, format="toml")  # type: ignore[arg-type]


def test_version_errors(config: ExperimentConfig) -> None:
    text = '{"schema_version":"2.0"}'
    with pytest.raises(UnsupportedConfigVersionError):
        parse_config(text, format="json")
    with pytest.raises(ValidationError):
        parse_config("{}", format="json")


def test_paths_and_roundtrip(config: ExperimentConfig, tmp_path: Path) -> None:
    for suffix in (".yaml", ".json"):
        path = tmp_path / f"c{suffix}"
        dump_config(config, path)
        assert load_config(path) == config
        with pytest.raises(ConfigIOError):
            dump_config(config, path)
        dump_config(config, path, overwrite=True)
    with pytest.raises(ConfigFormatError):
        load_config(tmp_path / "x.toml")
    with pytest.raises(ConfigIOError):
        load_config(tmp_path / "missing.yaml")
    with pytest.raises(ConfigIOError):
        dump_config(config, tmp_path / "missing" / "x.yaml")


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_serialization_revalidates_unchecked_model_copy(
    config: ExperimentConfig, tmp_path: Path, suffix: str
) -> None:
    unchecked = config.model_copy(update={"seed": -1})
    target = tmp_path / f"invalid{suffix}"

    with pytest.raises(ValidationError, match="seed"):
        serialize_config(unchecked, format="yaml" if suffix == ".yaml" else "json")
    with pytest.raises(ValidationError, match="seed"):
        dump_config(unchecked, target)

    assert not target.exists()
