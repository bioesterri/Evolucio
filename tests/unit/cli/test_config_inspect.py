# ruff: noqa: ANN001, ANN201

import json

import pytest

from evolucio.cli.main import main
from evolucio.config import freeze_config, load_config


@pytest.mark.parametrize("format", ["text", "json"])
def test_stable_normalized_inspection(config_path, config_data, capsys, format):
    argv = ["config", "inspect", str(config_path), "--format", format]
    assert main(argv) == 0
    first = capsys.readouterr()
    config_path.write_text(json.dumps(dict(reversed(list(config_data.items())))))
    assert main(argv) == 0
    second = capsys.readouterr()
    assert first == second
    assert not first.err
    assert freeze_config(load_config(config_path)).config_hash in first.out
    assert "Array(" not in first.out
    assert "0x" not in first.out
    if format == "json":
        data = json.loads(first.out)
        assert data["configuration"]["seed"] == 42
        assert data["schema_version"] == "3.0"
        assert data["static_compile_contract"]["world_width"] == 3
        assert data["configuration"]["policy"]["hidden_size"] == 16


def test_format_rejected(config_path, capsys):
    assert main(["config", "inspect", str(config_path), "--format", "yaml"]) == 2
    assert capsys.readouterr().err
