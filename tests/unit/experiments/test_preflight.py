# ruff: noqa: ANN001, ANN201

import subprocess
import sys
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from evolucio.config import freeze_config, load_config
from evolucio.experiments import (
    BatchDefinitionError,
    BatchPlan,
    ExperimentCase,
    SeedSet,
    preflight_batch,
    run_batch,
)


def test_preflight_loads_once_and_preserves_base(config_path, monkeypatch):
    base = load_config(config_path)
    loader = Mock(return_value=base)
    monkeypatch.setattr("evolucio.experiments.preflight.load_config", loader)
    before = freeze_config(base)
    view = preflight_batch(BatchPlan((ExperimentCase("A", config_path),), SeedSet((22, 11)), 4, 2))
    loader.assert_called_once_with(config_path)
    assert (view.case_count, view.seed_count, view.run_count) == (1, 2, 2)
    assert view.seeds == (22, 11)
    assert view.cases[0].config_hash == before.config_hash
    assert freeze_config(base) == before
    for seed, digest in view.cases[0].run_config_hashes:
        data = base.model_dump()
        data["seed"] = seed
        assert digest == freeze_config(type(base).model_validate(data)).config_hash


def test_identical_cases_share_hash(config_path):
    view = preflight_batch(
        BatchPlan(
            (ExperimentCase("A", config_path), ExperimentCase("B", config_path)),
            SeedSet((1,)),
            2,
            1,
        )
    )
    assert view.cases[0].config_hash == view.cases[1].config_hash
    assert view.cases[0].run_config_hashes == view.cases[1].run_config_hashes


@pytest.mark.parametrize("continue_on_error", [True, False])
def test_invalid_later_config_never_starts(config_path, tmp_path, monkeypatch, continue_on_error):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}")
    init = Mock()
    factory = Mock()
    monkeypatch.setattr("evolucio.runtime.initialization.initialize_simulation", init)
    plan = BatchPlan(
        (ExperimentCase("A", config_path), ExperimentCase("B", invalid)),
        SeedSet((11, 22)),
        4,
        2,
        continue_on_error=continue_on_error,
    )
    with pytest.raises(ValidationError):
        run_batch(plan=plan, consumer_factory=factory)
    init.assert_not_called()
    factory.assert_not_called()


def test_removed_path_rechecked(config_path):
    plan = BatchPlan((ExperimentCase("A", config_path),), SeedSet((11,)), 2, 1)
    config_path.unlink()
    with pytest.raises(BatchDefinitionError):
        run_batch(plan=plan)


def test_compile_invalid_later_config_never_starts(config_path, tmp_path, monkeypatch):
    from evolucio.config import dump_config

    data = load_config(config_path).model_dump()
    data["energy"]["basal_cost"] = 1e-100
    invalid = tmp_path / "underflow.json"
    dump_config(type(load_config(config_path)).model_validate(data), invalid)
    init = Mock()
    monkeypatch.setattr("evolucio.runtime.initialization.initialize_simulation", init)
    plan = BatchPlan(
        (ExperimentCase("A", config_path), ExperimentCase("B", invalid)), SeedSet((1,)), 2, 1
    )
    with pytest.raises(BatchDefinitionError, match="underflows"):
        run_batch(plan=plan)
    init.assert_not_called()


def test_host_preflight_imports_no_device_libraries(config_path):
    script = """
import sys
from pathlib import Path
from evolucio.experiments import *
p = BatchPlan((ExperimentCase("A", Path(sys.argv[1])),), SeedSet((1, 2)), 4, 2)
assert preflight_batch(p).run_count == 2
assert "jax" not in sys.modules
assert "equinox" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script, str(config_path)], check=True)
