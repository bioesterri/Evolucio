"""Use real persisted runs to check that byte integrity and semantics are independent."""

import json
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import polars as pl
import pytest
from typer.testing import CliRunner

from evolucio.analysis.summaries import AGENTS_PATH, LINEAGES_PATH
from evolucio.cli.main import app
from evolucio.persistence.loaders.compatibility import (
    check_read_compatibility,
    check_restore_compatibility,
    compare_runs,
)
from evolucio.persistence.loaders.errors import DataSchemaError, UnsupportedSchemaVersionError
from evolucio.persistence.loaders.run_data import open_run
from evolucio.persistence.local.artifacts import json_text, sha256_file
from evolucio.persistence.local.ownership import run_ownership
from evolucio.persistence.parquet.index import INDEX_PATH
from evolucio.persistence.validation.runner import validate_run
from evolucio.persistence.validation.types import CheckStatus, ValidationLevel

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "persistence"


@pytest.fixture(scope="module")
def persisted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("persisted")
    with ZipFile(FIXTURES / "recorded-run-pr41.zip") as archive:
        archive.extractall(root)
    (run_dir,) = root.iterdir()
    return run_dir


@pytest.fixture
def copied_run(persisted: Path, tmp_path: Path) -> Path:
    from shutil import copytree

    return copytree(persisted, tmp_path / persisted.name)


def test_scientific_validation_is_read_only(persisted: Path) -> None:
    def hashes() -> dict[Path, str]:
        return {path: sha256_file(path) for path in persisted.rglob("*") if path.is_file()}

    before = hashes()
    run = open_run(persisted)
    assert isinstance(run.scan_metrics(), pl.LazyFrame)
    assert run.scan_births().collect().height > 0
    assert run.list_snapshots() and run.list_checkpoints()
    assert run.list_highlighted_genomes()
    report = validate_run(persisted, validation_level=ValidationLevel.SCIENTIFIC)
    assert report.validation_passed, report.issues
    assert report.components["postgresql"] is CheckStatus.NOT_CHECKED
    assert report.components["mlflow"] is CheckStatus.NOT_CHECKED
    assert before == hashes()


def rewrite_part(run_dir: Path, dataset: str, expression: pl.Expr) -> None:
    index_file = run_dir / INDEX_PATH
    index = json.loads(index_file.read_bytes())
    part_record = index["datasets"][dataset]["parts"][0]
    part = run_dir / part_record["relative_path"]
    pl.read_parquet(part).with_columns(expression).write_parquet(part)
    part_record.update(byte_size=part.stat().st_size, sha256=sha256_file(part))
    index_file.write_text(json_text(index))
    refresh_artifact(run_dir, INDEX_PATH)


def refresh_artifact(run_dir: Path, relative: str) -> None:
    manifest_file = run_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_bytes())
    artifact = next(a for a in manifest["artifacts"] if a["relative_path"] == relative)
    path = run_dir / relative
    artifact.update(byte_size=path.stat().st_size, sha256=sha256_file(path))
    manifest_file.write_text(json_text(manifest))


def test_consistent_hashes_do_not_prove_genealogy(copied_run: Path) -> None:
    rewrite_part(copied_run, "genealogy", pl.lit(999, dtype=pl.Int64).alias("parent_agent_id"))
    integrity = validate_run(copied_run, validation_level=ValidationLevel.INTEGRITY)
    assert integrity.validation_passed, integrity.issues
    scientific = validate_run(copied_run, validation_level=ValidationLevel.SCIENTIFIC)
    assert not scientific.validation_passed
    assert any(issue.code == "genealogy_projection" for issue in scientific.issues)


@pytest.mark.parametrize("version", [None, 999])
def test_loader_requires_known_schema(copied_run: Path, version: int | None) -> None:
    path = copied_run / INDEX_PATH
    index = json.loads(path.read_bytes())
    if version is None:
        del index["schema_version"]
    else:
        index["schema_version"] = version
    path.write_text(json_text(index))
    with pytest.raises(DataSchemaError if version is None else UnsupportedSchemaVersionError):
        open_run(copied_run)


def test_readability_and_analysis_compatibility_are_separate(persisted: Path) -> None:
    first = open_run(persisted)
    manifest = replace(first.manifest, schemas=replace(first.manifest.schemas, metrics="f" * 64))
    second = replace(first, manifest=manifest)
    result = compare_runs(first, second)
    assert result.readable
    assert result.analysis_compatible is False
    assert result.restore_compatible is None


def test_checkpoint_restore_is_explicit(persisted: Path) -> None:
    run = open_run(persisted)
    assert check_read_compatibility(persisted).readable
    assert check_restore_compatibility(run, -1).restore_compatible is None
    report = check_restore_compatibility(run, run.list_checkpoints()[-1].state_step)
    assert report.restore_compatible, report.incompatibilities
    assert report.analysis_compatible is None


def test_active_writer_blocks_expensive_validation(persisted: Path) -> None:
    with run_ownership(persisted) as writer:
        assert writer == "inactive"
        report = validate_run(persisted)
        assert report.validation_passed
        assert any(issue.code == "writer_active" for issue in report.issues)
        assert not validate_run(
            persisted, validation_level=ValidationLevel.SCIENTIFIC
        ).validation_passed


def test_cli_json_is_machine_readable(persisted: Path) -> None:
    result = CliRunner().invoke(app, ["data", "validate", str(persisted), "--format", "json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["schema_version"] == 1


@pytest.mark.parametrize(
    ("damage", "code"),
    [
        ("missing", "missing_artifact"),
        ("size", "artifact_integrity"),
        ("hash", "artifact_integrity"),
        ("orphan", "orphan_artifact"),
        ("config", "artifact_integrity"),
        ("genome", "genome_verification"),
        ("checkpoint", "capture_verification"),
    ],
)
def test_integrity_detects_corruption(copied_run: Path, damage: str, code: str) -> None:
    run = open_run(copied_run)
    assert run.parquet is not None
    part = copied_run / run.parquet.datasets["metrics"].parts[0].relative_path
    if damage == "missing":
        part.unlink()
    elif damage == "size":
        part.write_bytes(part.read_bytes() + b"extra bytes")
    elif damage == "hash":
        part.write_bytes(b"!" + part.read_bytes()[1:])
    elif damage == "orphan":
        part.with_name("orphan.parquet").write_bytes(part.read_bytes())
    elif damage == "genome":
        genome = copied_run / run.list_highlighted_genomes()[0].relative_path
        genome.write_bytes(b"invalid")
    elif damage == "checkpoint":
        (copied_run / run.list_checkpoints()[0].relative_path / "zarr.json").write_text("{}")
    else:
        config = copied_run / "config.json"
        config.write_bytes(config.read_bytes() + b"\n")
    report = validate_run(copied_run, validation_level=ValidationLevel.INTEGRITY)
    assert not report.validation_passed
    assert any(issue.code == code for issue in report.issues), report.issues


def test_symlink_fails_before_scanning(copied_run: Path, tmp_path: Path) -> None:
    run = open_run(copied_run)
    assert run.parquet is not None
    part = copied_run / run.parquet.datasets["metrics"].parts[0].relative_path
    outside = tmp_path / "outside.parquet"
    part.rename(outside)
    part.symlink_to(outside)
    with pytest.raises(DataSchemaError, match="symlink"):
        open_run(copied_run)


def test_static_recovery_provenance() -> None:
    from pydantic import TypeAdapter

    from evolucio.recovery.provenance import RecoveryProvenance

    record = TypeAdapter(RecoveryProvenance).validate_json(
        (FIXTURES / "recovery-provenance-pr41.json").read_bytes(), strict=True
    )
    assert record.schema_version == 1
    assert record.inherited_steps == 40
    assert record.root_run_id == record.source_run_id


@pytest.mark.parametrize(
    ("dataset", "expression", "code"),
    [
        ("metrics", pl.col("alive_count").cast(pl.Int32), "data_schema"),
        ("metrics", pl.lit(None, dtype=pl.Int64).alias("alive_count"), "data_schema"),
        ("metrics", pl.lit(7, dtype=pl.Int64).alias("alive_count"), "population_balance"),
        ("births", pl.lit(999, dtype=pl.Int64).alias("parent_genome_id"), "invalid_parent"),
        ("births", pl.lit(999, dtype=pl.Int64).alias("lineage_id"), "invalid_parent"),
        ("births", pl.lit(999, dtype=pl.Int64).alias("generation"), "invalid_parent"),
        ("deaths", pl.lit(999, dtype=pl.Int64).alias("agent_id"), "invalid_death_origin"),
    ],
)
def test_columnar_validation(
    copied_run: Path, dataset: str, expression: pl.Expr, code: str
) -> None:
    rewrite_part(copied_run, dataset, expression)
    if dataset == "births":
        rewrite_part(copied_run, "genealogy", expression)
    report = validate_run(copied_run, validation_level=ValidationLevel.SCIENTIFIC)
    assert not report.validation_passed
    assert any(issue.code == code for issue in report.issues), report.issues


def test_optional_components_are_explicit(persisted: Path) -> None:
    report = validate_run(
        persisted, validation_level=ValidationLevel.STRUCTURAL, components=("parquet",)
    )
    assert report.validation_passed
    assert report.components["parquet"] is CheckStatus.PASSED
    assert report.components["snapshots"] is CheckStatus.NOT_CHECKED
    assert report.components["summaries"] is CheckStatus.NOT_CHECKED


def test_diagnostic_death_can_be_read(copied_run: Path) -> None:
    from evolucio.core.codes import DeathCauseCode

    rewrite_part(copied_run, "deaths", pl.lit(float("nan"), dtype=pl.Float32).alias("energy"))
    rewrite_part(copied_run, "deaths", pl.lit(-5, dtype=pl.Int64).alias("position_x"))
    rewrite_part(
        copied_run,
        "deaths",
        pl.lit(int(DeathCauseCode.INVALID_STATE), dtype=pl.Int32).alias("cause_code"),
    )
    rewrite_part(
        copied_run,
        "metrics",
        pl.col("deaths_energy_depletion").alias("deaths_invalid_state"),
    )
    rewrite_part(
        copied_run,
        "metrics",
        pl.lit(0, dtype=pl.Int64).alias("deaths_energy_depletion"),
    )
    report = validate_run(
        copied_run, validation_level=ValidationLevel.SCIENTIFIC, components=("parquet",)
    )
    assert report.validation_passed, report.issues


def test_local_loading_uses_no_optional_services(persisted: Path) -> None:
    import subprocess
    import sys

    script = """
import sys
from pathlib import Path
from evolucio.persistence.loaders.run_data import open_run

run = open_run(Path(sys.argv[1]))
assert run.scan_metrics().collect().height == 3
assert 'sqlalchemy' not in sys.modules
assert 'mlflow' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script, str(persisted)], check=True)


@pytest.mark.parametrize("status", ["failed", "stopped", "failed_overflow"])
def test_partial_prefix_requires_explicit_opt_in(copied_run: Path, status: str) -> None:
    path = copied_run / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["artifacts"] = [
        item for item in manifest["artifacts"] if item["relative_path"] != "summary.json"
    ]
    (copied_run / "summary.json").unlink()
    if status == "stopped":
        manifest["status"] = "stopped"
        manifest["stop"] = {
            "reason": "test_stop",
            "requested_at_utc": None,
            "stopped_at_utc": manifest["finished_at_utc"],
            "ordered": True,
            "last_confirmed_state_step": 3,
            "last_committed_telemetry_step": 3,
            "last_valid_checkpoint_step": 3,
        }
    else:
        manifest["status"] = "failed"
        manifest["final_step"] = None
        manifest["error"] = {
            "error_type": "EventBufferIntegrityError" if status == "failed_overflow" else "OSError",
            "message": "test failure",
            "last_committed_telemetry_step": 3,
        }
    path.write_text(json_text(manifest))
    before = validate_run(copied_run, validation_level=ValidationLevel.SCIENTIFIC)
    assert not before.validation_passed
    assert before.data_complete is False
    after = validate_run(
        copied_run, validation_level=ValidationLevel.SCIENTIFIC, allow_partial=True
    )
    assert after.validation_passed is (status != "failed_overflow"), after.issues
    if status == "failed_overflow":
        assert any(issue.code == "event_buffer_loss" for issue in after.issues)


@pytest.mark.parametrize(
    ("path", "column", "code"),
    [
        (AGENTS_PATH, "direct_offspring_count", "agent_summary"),
        (LINEAGES_PATH, "total_descendants", "lineage_summary"),
    ],
)
def test_summary_projection_disagrees(copied_run: Path, path: str, column: str, code: str) -> None:
    file = copied_run / path
    metadata = pl.read_parquet_metadata(file)
    pl.read_parquet(file).with_columns(pl.col(column) + 1).write_parquet(
        file, metadata={key: metadata[key] for key in ("schema_name", "schema_version")}
    )
    refresh_artifact(copied_run, path)
    report = validate_run(copied_run, validation_level=ValidationLevel.SCIENTIFIC)
    assert not report.validation_passed
    assert any(issue.code == code for issue in report.issues), report.issues


def test_exact_metric_sampling(copied_run: Path) -> None:
    path = copied_run / INDEX_PATH
    index = json.loads(path.read_bytes())
    index["policy"]["metrics_interval"] = 2
    path.write_text(json_text(index))
    refresh_artifact(copied_run, INDEX_PATH)
    assert validate_run(copied_run, validation_level=ValidationLevel.STRUCTURAL).validation_passed
    scientific = validate_run(copied_run, validation_level=ValidationLevel.SCIENTIFIC)
    assert "data_consistency" in {issue.code for issue in scientific.issues}
    assert not scientific.validation_passed


def test_self_parent_cycle_is_detected(copied_run: Path) -> None:
    for dataset in ("births", "genealogy"):
        rewrite_part(copied_run, dataset, pl.col("child_agent_id").alias("parent_agent_id"))
    report = validate_run(copied_run, validation_level=ValidationLevel.SCIENTIFIC)
    assert any(issue.code == "invalid_parent" for issue in report.issues)
    assert not report.validation_passed


def test_completed_run_requires_registered_analysis_artifacts(copied_run: Path) -> None:
    path = copied_run / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["artifacts"] = [
        record for record in manifest["artifacts"] if record["relative_path"] != LINEAGES_PATH
    ]
    path.write_text(json_text(manifest))
    assert any(issue.code == "missing_artifact" for issue in validate_run(copied_run).issues)
