# ruff: noqa: ANN001, ANN201

import json

import pytest

from evolucio.cli.main import main


def test_valid(config_path, capsys):
    assert main(["config", "validate", str(config_path)]) == 0
    output = capsys.readouterr()
    assert "Configuration valid" in output.out
    assert "schema_version" in output.out
    assert "compile_signature" in output.out
    assert not output.err


@pytest.mark.parametrize(
    ("block", "field", "value"),
    [
        ("world", "width", 0),
        ("world", "height", -1),
        ("world", "width", "3"),
        ("energy", "initial_energy", -1),
        ("energy", "reproduction_cost", 0),
        ("energy", "reproduction_threshold", 0),
        ("evolution", "weight_mutation_rate", 1.2),
        ("evolution", "bias_mutation_rate", -0.1),
        ("world", "unknown", 1),
    ],
)
def test_invalid(config_path, config_data, block, field, value, capsys):
    config_data[block][field] = value
    config_path.write_text(json.dumps(config_data))
    assert main(["config", "validate", str(config_path)]) == 2
    output = capsys.readouterr()
    assert "Configuration invalid" in output.err
    assert not output.out
    assert "Traceback" not in output.err


@pytest.mark.parametrize("content", ["{", "{}", '{"seed": true}', "[]"])
def test_invalid_document(config_path, content, capsys):
    config_path.write_text(content)
    assert main(["config", "validate", str(config_path)]) == 2
    assert capsys.readouterr().err


def test_missing_seed(config_path, config_data, capsys):
    del config_data["seed"]
    config_path.write_text(json.dumps(config_data))
    assert main(["config", "validate", str(config_path)]) == 2
    assert "seed" in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["missing", "directory", "extension"])
def test_paths(tmp_path, kind, capsys):
    path = tmp_path / "missing.json"
    expected = "file does not exist"
    if kind == "directory":
        path = tmp_path
        expected = "not a file"
    elif kind == "extension":
        path = tmp_path / "file.toml"
        path.write_text("{}")
        expected = "unsupported configuration extension"
    assert main(["config", "validate", str(path)]) == 2
    assert expected in capsys.readouterr().err


def test_validate_never_initializes(config_path, monkeypatch):
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("validation initialized or executed")

    monkeypatch.setattr("evolucio.runtime.initialization.initialize_simulation", forbidden)
    monkeypatch.setattr("evolucio.runtime.run_simulation", forbidden)
    assert main(["config", "validate", str(config_path)]) == 0
