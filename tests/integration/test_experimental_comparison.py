"""Exercise planned execution and paired analysis on persisted real runs."""

import re
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from evolucio.analysis.comparison import LocalRunProvider, compare_experiment, export_comparison
from evolucio.analysis.comparison.metrics import extract_metrics
from evolucio.analysis.comparison.selection import RunCandidate, select_comparison_runs
from evolucio.analysis.comparison.temporal import compare_time_series
from evolucio.analysis.genome_distance import parent_child_genome_divergence
from evolucio.analysis.types import AnalysisWindow
from evolucio.cli.main import app
from evolucio.experiments.design import (
    ExperimentDesign,
    PairPolicy,
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
            "cases": full.cases[:3],
            "allowed_config_differences": {
                case.case_id: full.allowed_config_differences[case.case_id]
                for case in full.cases[1:3]
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
    design_path = tmp_path / "design.json"
    design_path.write_text(design.model_dump_json(), encoding="utf-8")
    execution = CliRunner().invoke(
        app, ["experiment", "execute-design", str(design_path), "--execution-dir", str(root)]
    )
    assert execution.exit_code == 0, execution.output
    assert "9/9 completats" in execution.output
    provider = LocalRunProvider(root, planned_execution=True)
    result = compare_experiment(design, run_provider=provider)
    assert result.interpretation == "PLANNED"
    assert all(
        report.comparison_eligible and report.paired_seeds == (11, 22, 33)
        for report in result.comparability_reports
    )
    assert result.paired_difference_summaries.filter(
        pl.col("metric_name") == "final_population"
    ).collect()["n_pairs"].to_list() == [3, 3]
    control = result.tables["run_metrics"].filter(
        (pl.col("case_id") == "no_mutation") & (pl.col("metric_name") == "mutation_effective_total")
    )
    assert control.collect()["metric_value"].to_list() == [0, 0, 0]
    baseline = result.run_selection_report.selected[("baseline", 11)]
    observed = extract_metrics(
        baseline,
        design.analysis_policy,
        (
            "bottleneck_observed_duration",
            "survival_at_fixed_age",
            "deaths_by_cause",
            "death_fraction_by_cause",
            "within_lineage_variance",
            "founder_genome_divergence",
        ),
        fixed_age=0,
    )
    values = {(item.name, item.dimension): item.value for item in observed}
    assert values["bottleneck_observed_duration", ""] == 0
    assert values["survival_at_fixed_age", ""] == 1
    assert values["within_lineage_variance", ""] is None
    assert values["founder_genome_divergence", ""] is not None
    assert all(item.value == 0 for item in observed if item.name == "deaths_by_cause")
    assert all(item.value is None for item in observed if item.name == "death_fraction_by_cause")
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
    cli_output = tmp_path / "cli-comparison"
    comparison = CliRunner().invoke(
        app,
        [
            "analysis",
            "compare",
            str(design_path),
            "--execution-dir",
            str(root),
            "--output-dir",
            str(cli_output),
            "--bootstrap-seed",
            "45",
            "--temporal-metric",
            "population_after",
            "--environment-control",
            "constant_environment",
            "--common-steps",
        ],
    )
    assert comparison.exit_code == 0, (comparison.output, comparison.exception)
    assert "EXPLORATORY" in comparison.output
    assert pl.read_parquet(cli_output / "differential_change.parquet").sort("seed").select(
        "seed", "differential_change"
    ).rows() == [(11, 0.0), (22, 0.0), (33, 0.0)]
    paired_time = pl.read_parquet(cli_output / "paired_time_series.parquet")
    assert paired_time.height == 2 * design.total_steps
    assert paired_time["n_pairs_available"].to_list() == [3] * paired_time.height
    interval = pl.read_parquet(cli_output / "paired_summaries.parquet").filter(
        (pl.col("variant_case_id") == "no_mutation") & (pl.col("metric_name") == "final_population")
    )
    assert interval.select("bootstrap_low", "bootstrap_high").row(0) == (0.0, 0.0)
    assert all(sha256_file(path) == digest for path, digest in original.items())
    temporal = compare_time_series(
        result.run_selection_report.selected,
        "population_after",
        AnalysisWindow(0, design.total_steps),
    ).collect()
    assert temporal.height == 3 * design.total_steps
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
    assert not next(
        item for item in strict.comparability_reports if item.variant_case_id == "no_mutation"
    ).comparison_eligible
    available = compare_experiment(
        design, run_provider=missing_control, comparison_policy=PairPolicy.AVAILABLE_PAIRS
    )
    partial = next(
        item for item in available.comparability_reports if item.variant_case_id == "no_mutation"
    )
    assert partial.paired_seeds == (11, 33)
    assert partial.missing_seeds == (22,)
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
    recovery_output = tmp_path / "chosen-comparison"
    recovery_comparison = CliRunner().invoke(
        app,
        [
            "analysis",
            "compare",
            str(design_path),
            "--runs-root",
            str(root / "runs"),
            "--select-run",
            f"baseline:11={recovered.manifest.run_id}",
            "--allow-available-pairs",
            "--output-dir",
            str(recovery_output),
        ],
    )
    assert recovery_comparison.exit_code == 0, recovery_comparison.output
    recovered_ids = (
        pl.read_parquet(recovery_output / "paired_differences.parquet")
        .filter(pl.col("seed") == 11)["baseline_run_id"]
        .unique()
        .to_list()
    )
    assert recovered_ids == [recovered.manifest.run_id]
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


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--execution-dir", ".", "--registered"], "--execution-dir is exclusive"),
        (["--environment-control", "constant_environment"], "requires --temporal-metric"),
        (["--select-run", "baseline"], "requires CASE:SEED=RUN_ID"),
        (
            ["--select-run", "baseline:11=a", "--select-run", "baseline:11=b"],
            "duplicate explicit selection",
        ),
    ],
)
def test_comparison_cli_rejects_invalid_options(options: list[str], message: str) -> None:
    result = CliRunner().invoke(app, ["analysis", "compare", str(EXAMPLE), *options], color=True)
    output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert result.exit_code == 2, output
    assert message in output


def test_comparison_cli_exports_missing_seed_audit(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    result = CliRunner().invoke(
        app,
        [
            "analysis",
            "compare",
            str(EXAMPLE),
            "--runs-root",
            str(tmp_path / "missing"),
            "--output-dir",
            str(output),
            "--temporal-metric",
            "population_after",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "STRICT requires every planned seed pair" in result.output
    coverage = pl.read_parquet(output / "seed_coverage.parquet")
    assert coverage["missing_seed_count"].sum() == 12
    assert coverage["eligible_seed_count"].sum() == 0
    assert not (output / "time_series_summary.parquet").exists()


def test_validate_design_cli_reports_uncontrolled_factor(tmp_path: Path) -> None:
    design = load_design(EXAMPLE)
    design = design.model_copy(
        update={
            "allowed_config_differences": {
                **design.allowed_config_differences,
                "no_mutation": (),
            },
        }
    )
    path = tmp_path / "blocked-design.json"
    path.write_text(design.model_dump_json(), encoding="utf-8")
    result = CliRunner().invoke(app, ["experiment", "validate-design", str(path)])
    assert result.exit_code == 2, result.output
    assert "Uncontrolled parameter: evolution.weight_mutation_rate" in result.output
