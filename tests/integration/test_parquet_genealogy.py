# ruff: noqa: ANN001, ANN201
from dataclasses import replace

from evolucio.config import load_config
from evolucio.persistence.local import create_local_run
from evolucio.persistence.parquet import ParquetRunConsumer, scan_genealogy
from evolucio.runtime import RunMode, RunPlan
from tests.unit.persistence.parquet.conftest import birth, chunk


def test_reconstruct_ancestry_without_state(tmp_path):
    recorder = create_local_run(
        config=load_config("tests/fixtures/config/valid_v1.json"),
        plan=RunPlan(20, 4),
        mode=RunMode.EXPERIMENT,
        output_root=tmp_path,
    )
    sink = ParquetRunConsumer(recorder)
    sink(
        chunk(
            births=[
                birth(1, 2, 1),
                birth(2, 3, 1),
                replace(birth(3, 4, 2), generation=2),
                replace(birth(4, 5, 4), generation=3),
            ]
        )
    )
    sink.finalize()
    parents = dict(
        scan_genealogy(recorder.run_dir)
        .collect()
        .select("child_agent_id", "parent_agent_id")
        .iter_rows()
    )
    chain = [5]
    while chain[-1] in parents:
        chain.append(parents[chain[-1]])
    assert chain == [5, 4, 2, 1]
