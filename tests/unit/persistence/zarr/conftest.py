# ruff: noqa: ANN001, ANN201
from uuid import uuid4

import pytest

from evolucio.config import compile_config, load_config
from evolucio.config.compile import compile_signature_digest
from evolucio.core.metrics import create_empty_metrics_accumulator
from evolucio.persistence.local import create_local_run
from evolucio.persistence.zarr.checkpoint import write_checkpoint
from evolucio.runtime import RunMode, RunPlan
from tests.regression.helpers import canonical_case


@pytest.fixture
def carry():
    state, config, digest = canonical_case()
    return state, create_empty_metrics_accumulator(), config, digest


@pytest.fixture
def checkpoint(tmp_path, carry):
    state, accumulator, config, digest = carry
    record = write_checkpoint(
        run_dir=tmp_path,
        run_id=str(uuid4()),
        state=state,
        accumulator=accumulator,
        config=config,
        config_hash=digest,
        compile_signature="a" * 64,
    )
    return tmp_path / record.relative_path


@pytest.fixture
def recorder(tmp_path):
    config = load_config("tests/fixtures/config/valid_v1.json")
    compiled = compile_config(config)
    return create_local_run(
        config=config,
        plan=RunPlan(30, 7),
        mode=RunMode.EXPERIMENT,
        output_root=tmp_path,
        compile_signature=compile_signature_digest(compiled.compile_signature),
    )
