# ruff: noqa: ANN001, ANN201

import pytest

from evolucio.cli.formatting import json_text, text_fields
from evolucio.cli.main import main


def test_formatting():
    assert text_fields({"flag": True, "value": 0.125}) == "flag: true\nvalue: 0.125"
    assert json_text({"b": 1, "a": None}).index('"a"') < json_text({"b": 1, "a": None}).index('"b"')
    with pytest.raises(ValueError):
        json_text(float("nan"))


@pytest.mark.parametrize(
    "command", [[], ["config"], ["config", "validate"], ["config", "inspect"], ["run"]]
)
def test_help(command, capsys):
    assert main([*command, "--help"]) == 0
    assert "Usage" in capsys.readouterr().out


def test_version(capsys):
    from evolucio import __version__

    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__
