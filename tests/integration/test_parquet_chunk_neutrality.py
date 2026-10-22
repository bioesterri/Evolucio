# ruff: noqa: ANN001, ANN201
from evolucio.config import load_config
from evolucio.persistence.parquet import ParquetPersistencePolicy
from evolucio.persistence.parquet.reader import scan_dataset
from evolucio.runtime import RunPlan
from evolucio.runtime.recorded_run import run_recorded_simulation
from tests.experiment_helpers import write_cases


def test_real_chunk_neutrality(tmp_path):
    config = load_config(write_cases(tmp_path)[0].config_path)
    runs = [
        run_recorded_simulation(
            config=config,
            plan=RunPlan(20, size),
            parquet_policy=ParquetPersistencePolicy(metrics_interval=5),
            output_root=tmp_path / "runs",
        )
        for size in (4, 7)
    ]
    for name in ("metrics", "births", "deaths", "genealogy"):
        frames = [scan_dataset(run.run_dir, name).drop("run_id").collect() for run in runs]
        if name == "metrics":
            assert frames[0]["step"].to_list() == frames[1]["step"].to_list() == [0, 5, 10, 15]
            # Floating reductions may differ at machine precision between JIT chunk shapes.
            from polars.testing import assert_frame_equal

            assert_frame_equal(frames[0], frames[1], rel_tol=1e-5, abs_tol=1e-5)
        else:
            assert frames[0].equals(frames[1])
