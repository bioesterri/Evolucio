# ruff: noqa: ANN001, ANN201

from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import pytest

from evolucio.config import load_config
from evolucio.config.compile import compile_config
from evolucio.runtime.initialization import InitializationError, initialize_simulation


def test_initialization_diagnostics(monkeypatch):
    config = compile_config(load_config("tests/fixtures/config/valid_v1.json")).core
    monkeypatch.setattr("evolucio.runtime.initialization.initialize_world", Mock())
    founders = SimpleNamespace(overflow=True)
    monkeypatch.setattr(
        "evolucio.runtime.initialization.initialize_population", Mock(return_value=founders)
    )
    with pytest.raises(InitializationError, match="overflow"):
        initialize_simulation(config, seed=42)
    founders.overflow = False
    founders.population = SimpleNamespace(alive=jnp.array([True]), genome_id=jnp.array([-1]))
    monkeypatch.setattr(
        "evolucio.runtime.initialization.initialize_genome_batch",
        Mock(return_value=SimpleNamespace(invalid_active_genome_id_count=1)),
    )
    with pytest.raises(InitializationError, match="genome identity"):
        initialize_simulation(config, seed=42)
