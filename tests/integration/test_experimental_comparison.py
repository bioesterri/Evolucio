"""Exercise planned execution and paired analysis on persisted real runs."""

from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from evolucio.analysis.comparison import LocalRunProvider, compare_experiment, export_comparison
from evolucio.analysis.comparison.selection import RunCandidate, select_comparison_runs
from evolucio.analysis.comparison.temporal import compare_time_series
from evolucio.analysis.genome_distance import parent_child_genome_divergence
from evolucio.analysis.types import AnalysisWindow
from evolucio.cli.main import app
from evolucio.experiments.design import (
    ExperimentDesign,
    PairPolicy,
    execute_design,
    load_design,
    resolve_design,
)
from evolucio.persistence.local.artifacts import sha256_file
from evolucio.persistence.local.manifest import RunStatus, load_run_manifest
from evolucio.persistence.local.version_info import CodeVersionInfo
from evolucio.persistence.zarr.schemas import ZarrCapturePolicy
from evolucio.recovery.runner import resume
from evolucio.runtime.recorded_run import RecordedRunResult, run_recorded_simulation
from evolucio.runtime.types import HostChunkBatch, RunPlan
from tests.regression.helpers import FLOAT32_ATOL, FLOAT32_RTOL
from tests.unit.test_experimental_comparison import EXAMPLE


@pytest.fixture
def clean_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = CodeVersionInfo("0.1.0.dev0", "a" * 40, False)

    def version() -> CodeVersionInfo:
        return identity

    monkeypatch.setattr("evolucio.persistence.local.version_info.code_version_info", version)
    monkeypatch.setattr("evolucio.recovery.assessment.code_version_info", version)


def test_planned_comparison_end_to_end(tmp_path: Path, clean_provenance: None) -> None:
    full = load_design(EXAMPLE)
    design = full.model_copy(
        update={
            "cases": full.cases[:2],
            "allowed_config_differences": {
                "no_mutation": full.allowed_config_differences["no_mutation"]
            },
        }
    )
    dry_path = tmp_path / "dry"
    dry = CliRunner().invoke(
        app,
        [
            "experiment",
            "execute-design",
            str(EXAMPLE),
            "--dry-run",
            "--execution-dir",
            str(dry_path),
        ],
    )
    assert dry.exit_code == 0, dry.output
    assert not dry_path.exists()
    root = tmp_path / "experiment"
    resolved = resolve_design(design)
    batch = execute_design(resolved, root)
    assert batch.completed_run_count == 6
    provider = LocalRunProvider(root, planned_execution=True)
    result = compare_experiment(design, run_provider=provider)
    assert result.interpretation == "PLANNED"
    (report,) = result.comparability_reports
    assert report.comparison_eligible
    assert report.paired_seeds == (11, 22, 33)
    assert result.paired_difference_summaries.filter(
        pl.col("metric_name") == "final_population"
    ).collect()["n_pairs"].to_list() == [3]
    control = result.tables["run_metrics"].filter(
        (pl.col("case_id") == "no_mutation") & (pl.col("metric_name") == "mutation_effective_total")
    )
    assert control.collect()["metric_value"].to_list() == [0, 0, 0]
    verified_children = 0
    for (case, _), run in result.run_selection_report.selected.items():
        if case != "no_mutation":
            continue
        for child in run.run.scan_births().collect()["child_agent_id"].to_list():
            comparison = parent_child_genome_divergence(run, child)
            if comparison["quality"] == "EXACT":
                verified_children += 1
                assert comparison["rms_parameter_difference"] == 0
    assert verified_children > 0
    original = {path: sha256_file(path) for path in (root / "runs").rglob("*") if path.is_file()}
    output = export_comparison(result, tmp_path / "comparison")
    assert (output / "manifest.json").is_file()
    assert all(sha256_file(path) == digest for path, digest in original.items())
    temporal = compare_time_series(
        result.run_selection_report.selected,
        "population_after",
        AnalysisWindow(0, design.total_steps),
    ).collect()
    assert temporal.height == 2 * design.total_steps
    assert temporal["n_available_runs"].min() == 3

    def missing_control(definition: ExperimentDesign) -> Iterable[RunCandidate]:
        return reversed(
            [
                candidate
                for candidate in provider(definition)
                if (candidate.case_id, candidate.seed) != ("no_mutation", 22)
            ]
        )

    strict = compare_experiment(design, run_provider=missing_control)
    assert not strict.comparability_reports[0].comparison_eligible
    available = compare_experiment(
        design, run_provider=missing_control, comparison_policy=PairPolicy.AVAILABLE_PAIRS
    )
    assert available.comparability_reports[0].paired_seeds == (11, 33)
    assert available.comparability_reports[0].missing_seeds == (22,)
    assert available.interpretation == "EXPLORATORY"

    # Fail after a committed checkpoint, then recover the same root trajectory.
    failed_run_id = str(uuid4())
    failed_path = root / "runs" / failed_run_id

    def fail_after_checkpoint(chunk: HostChunkBatch) -> None:
        if chunk.start_step >= design.chunk_steps:
            raise RuntimeError("synthetic technical failure")

    config = resolved.batch.configs[0].config
    with pytest.raises(RuntimeError, match="synthetic technical failure"):
        run_recorded_simulation(
            config=config,
            case_id="baseline",
            output_root=root / "runs",
            run_id=failed_run_id,
            plan=RunPlan(design.total_steps, design.chunk_steps),
            capture_policy=ZarrCapturePolicy(checkpoint_interval=design.chunk_steps),
            consumer=fail_after_checkpoint,
        )
    assert load_run_manifest(failed_path / "manifest.json").status is RunStatus.FAILED
    recovered = resume(failed_path)
    assert isinstance(recovered, RecordedRunResult)
    ambiguous = select_comparison_runs(resolved, provider)
    assert ambiguous.cells[("baseline", 11)] == "ambiguous"
    failed = next(item for item in ambiguous.attempts if item.run_id == failed_run_id)
    assert failed.status == "failed" and failed.reason == "synthetic technical failure"
    chosen = compare_experiment(
        design, run_provider=provider, choices={("baseline", 11): recovered.manifest.run_id}
    )
    selected = chosen.run_selection_report.selected[("baseline", 11)].run.recovery
    assert selected is not None and selected.root_run_id == failed_run_id
    original_pairs = result.tables["paired_differences"].collect().drop("baseline_run_id")
    recovered_pairs = chosen.tables["paired_differences"].collect().drop("baseline_run_id")
    assert_frame_equal(
        original_pairs.drop("absolute_difference"),
        recovered_pairs.drop("absolute_difference"),
        abs_tol=FLOAT32_ATOL,
        rel_tol=FLOAT32_RTOL,
    )
    # Subtraction can expose the numerical error of either operand near delta=0.
    for original, recovered_row in zip(
        original_pairs.iter_rows(named=True), recovered_pairs.iter_rows(named=True), strict=True
    ):
        delta = original["absolute_difference"]
        restored_delta = recovered_row["absolute_difference"]
        if delta is None:
            assert restored_delta is None
        else:
            bound = 2 * FLOAT32_ATOL + FLOAT32_RTOL * (
                abs(original["baseline_value"]) + abs(original["variant_value"])
            )
            assert restored_delta == pytest.approx(delta, abs=bound, rel=0)


def test_sampled_energy_and_extinct_population(tmp_path: Path, clean_provenance: None) -> None:
    from evolucio.analysis.comparison.temporal import AlignmentPolicy
    from evolucio.config import load_config
    from evolucio.persistence.parquet.schemas import ParquetPersistencePolicy

    original = load_design(EXAMPLE)
    config = load_config(original.cases[0].config_path)
    config = config.model_copy(
        update={
            "seed": 11,
            "population": config.population.model_copy(update={"initial_agents": 0}),
        }
    )
    path = tmp_path / "empty.json"
    path.write_text(config.model_dump_json(), encoding="utf-8")
    design = original.model_copy(
        update={
            "cases": tuple(
                case.model_copy(update={"config_path": path}) for case in original.cases[:2]
            ),
            "seed_set": (11,),
            "environment_change_step": None,
            "comparison_metrics": original.comparison_metrics.model_copy(
                update={"primary": ("final_population",), "secondary": ("resource_consumed_total",)}
            ),
            "allowed_config_differences": {"no_mutation": ()},
        }
    )
    for case, chunk, interval in (("baseline", 3, 1), ("no_mutation", 2, 2)):
        run_recorded_simulation(
            config=config,
            case_id=case,
            output_root=tmp_path,
            plan=RunPlan(design.total_steps, chunk),
            parquet_policy=ParquetPersistencePolicy(metrics_interval=interval),
        )
    result = compare_experiment(design, run_provider=LocalRunProvider(tmp_path))
    (report,) = result.comparability_reports
    assert report.comparison_eligible and report.paired_seeds == (11,)
    assert report.coverage_differences
    assert any(diff.parameter_path == "manifest.chunk_steps" for diff in report.config_differences)
    data = (
        result.tables["run_metrics"]
        .filter(
            (pl.col("case_id") == "no_mutation")
            & (pl.col("metric_name") == "resource_consumed_total")
        )
        .collect()
        .row(0, named=True)
    )
    assert data["metric_value"] is None and data["metric_quality"] == "UNAVAILABLE"
    pair = (
        result.tables["paired_differences"]
        .filter(pl.col("metric_name") == "final_population")
        .collect()
        .row(0, named=True)
    )
    assert pair["absolute_difference"] == 0 and pair["relative_difference"] is None
    runs = result.run_selection_report.selected
    window = AnalysisWindow(0, design.total_steps)
    population = compare_time_series(runs, "population_after", window).collect()
    assert population["mean"].to_list() == [0] * (2 * design.total_steps)
    assert population["n_available_runs"].to_list() == [1] * (2 * design.total_steps)
    with pytest.raises(ValueError, match="observation steps differ"):
        compare_time_series(runs, "total_resources_available", window)
    energy = compare_time_series(
        runs, "total_resources_available", window, AlignmentPolicy.COMMON_STEPS
    ).collect()
    assert energy["step"].unique().sort().to_list() == [0, 2, 4]
    assert energy["n_available_runs"].to_list() == [1] * 6
