"""Tests for the installed package contract."""

from __future__ import annotations

import importlib
import sys
import tomllib
from importlib import metadata
from pathlib import Path


def test_package_metadata_matches_project() -> None:
    """The importable package and distribution expose the declared identity."""
    package = importlib.import_module("evolucio")
    distribution = metadata.distribution("projecte-evolucio")
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert package.__name__ == "evolucio"
    assert distribution.metadata["Name"] == "projecte-evolucio"
    assert distribution.version == project["version"]
    assert package.__version__ == distribution.version


def test_package_exposes_no_future_simulation_api_or_optional_layers() -> None:
    """Importing the root package stays independent from future and optional layers."""
    optional_roots = {"mlflow", "polars", "sqlalchemy", "matplotlib"}
    loaded_before = optional_roots.intersection(sys.modules)

    package = importlib.reload(importlib.import_module("evolucio"))

    assert not hasattr(package, "SimulationState")
    assert not hasattr(package, "simulate")
    assert optional_roots.intersection(sys.modules) == loaded_before
