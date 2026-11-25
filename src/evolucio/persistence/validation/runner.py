"""Orchestrate cumulative local checks and optional read-only reconciliation."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import polars as pl

from evolucio.analysis.summaries import AGENTS_PATH, FOUNDERS_PATH, LINEAGES_PATH
from evolucio.persistence.genomes import INDEX_PATH as GENOME_PATH
from evolucio.persistence.genomes import verify_genome_store
from evolucio.persistence.loaders.errors import (
    ArtifactIntegrityError,
    DataConsistencyError,
    UnsupportedSchemaVersionError,
)
from evolucio.persistence.loaders.run_data import RunData, read_run_metadata, registered_run_path
from evolucio.persistence.local.artifacts import artifact_path
from evolucio.persistence.local.manifest import RunStatus
from evolucio.persistence.local.ownership import run_ownership
from evolucio.persistence.local.verification import verify_local_run
from evolucio.persistence.parquet.index import INDEX_PATH as PARQUET_PATH
from evolucio.persistence.parquet.verify import verify_parquet_datasets
from evolucio.persistence.zarr.index import INDEX_PATH as ZARR_PATH
from evolucio.persistence.zarr.integrity import verify_zarr_artifacts
from evolucio.recovery.provenance import PROVENANCE_PATH

from .captures import check_captures
from .recovery import check_recovery
from .scientific import check_population, check_summaries, check_telemetry
from .structural import check_analysis, check_chunks
from .types import (
    CheckStatus,
    ValidationIssue,
    ValidationLevel,
    ValidationReport,
    ValidationSeverity,
)

if TYPE_CHECKING:
    from evolucio.persistence.postgres.repository import PostgresRegistry

COMPONENTS = (
    "manifest",
    "parquet",
    "population",
    "snapshots",
    "checkpoints",
    "genomes",
    "summaries",
    "recovery",
)


_ARTIFACTS_BY_COMPONENT = {
    "parquet": (PARQUET_PATH,),
    "population": (FOUNDERS_PATH,),
    "snapshots": (ZARR_PATH,),
    "checkpoints": (ZARR_PATH,),
    "genomes": (GENOME_PATH,),
    "summaries": ("summary.json", AGENTS_PATH, LINEAGES_PATH, FOUNDERS_PATH),
    "recovery": (PROVENANCE_PATH,),
}


def _code(error: Exception) -> str:
    if isinstance(error, UnsupportedSchemaVersionError):
        return "unsupported_schema_version"
    if isinstance(error, FileNotFoundError):
        return "missing_artifact"
    if isinstance(error, ArtifactIntegrityError):
        return "artifact_integrity"
    if isinstance(error, DataConsistencyError):
        return "data_consistency"
    return "data_schema"


def evaluate_run(
    run_dir: Path,
    *,
    validation_level: ValidationLevel = ValidationLevel.METADATA,
    components: Iterable[str] = COMPONENTS,
    registry: "PostgresRegistry | None" = None,
    tracking_uri: str | None = None,
    allow_partial: bool = False,
) -> tuple[RunData | None, ValidationReport, Exception | None]:
    selected = set(components)
    if not selected or selected.difference(COMPONENTS):
        raise ValueError("select one or more known local components")
    if validation_level is ValidationLevel.SCIENTIFIC and selected.intersection(
        ("population", "summaries", "snapshots", "checkpoints", "recovery")
    ):
        selected.add("parquet")
    statuses: dict[str, CheckStatus] = {
        name: CheckStatus.NOT_CHECKED
        for name in (
            *COMPONENTS,
            "recovery_source",
            "telemetry_coverage",
            "snapshot_metrics",
            "checkpoint_metrics",
            "postgresql",
            "mlflow",
            "tracking_metadata",
            "configuration",
        )
    }
    issues: list[ValidationIssue] = []
    data = None
    error = None
    blocked = False
    structural = validation_level.includes(ValidationLevel.STRUCTURAL)
    scientific = validation_level.includes(ValidationLevel.SCIENTIFIC)

    def finding(
        code: str,
        component: str,
        message: str,
        count: int | None = None,
        *,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        path: str | None = None,
    ) -> None:
        if len([item for item in issues if item.component == component and item.code == code]) < 20:
            issues.append(
                ValidationIssue(code, severity, component, message, path, violation_count=count)
            )
        if severity is ValidationSeverity.ERROR:
            statuses[component] = CheckStatus.FAILED

    def check(component: str, action: Callable[[], object]) -> None:
        if statuses[component] is not CheckStatus.FAILED:
            statuses[component] = CheckStatus.PASSED
        try:
            action()
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            finding(_code(exc), component, str(exc))

    with run_ownership(run_dir) as writer:
        try:
            data = read_run_metadata(run_dir)
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            error = exc
            if writer == "active":
                finding(
                    "writer_active",
                    "manifest",
                    "Metadata was unavailable during an active write",
                    severity=ValidationSeverity.WARNING,
                )
            else:
                finding(_code(exc), "manifest", str(exc))
        if data is not None:
            run = data
            statuses["manifest"] = CheckStatus.PASSED
            statuses["configuration"] = CheckStatus.PASSED
            if run.tracking is not None:
                statuses["tracking_metadata"] = CheckStatus.PASSED
            elif run.tracking_metadata_error is not None:
                finding(
                    "tracking_metadata_invalid",
                    "tracking_metadata",
                    run.tracking_metadata_error,
                    severity=ValidationSeverity.WARNING,
                )
            else:
                statuses["tracking_metadata"] = CheckStatus.NOT_APPLICABLE
            failure = run.manifest.error
            if (
                failure is not None
                and validation_level.includes(ValidationLevel.INTEGRITY)
                and (
                    failure.error_type == "EventBufferIntegrityError"
                    or failure.message == "death event count or overflow is inconsistent"
                )
            ):
                finding("event_buffer_loss", "manifest", failure.message)

            active = writer == "active" or (
                writer == "indeterminate"
                and run.manifest.status in (RunStatus.CREATED, RunStatus.RUNNING)
            )
            if active:
                finding(
                    "writer_active",
                    "manifest",
                    "Writer ownership prevents a stable audit",
                    severity=ValidationSeverity.WARNING,
                )
                blocked = validation_level is not ValidationLevel.METADATA
            elif validation_level.includes(ValidationLevel.INTEGRITY):

                def local() -> None:
                    paths = (
                        None
                        if selected == set(COMPONENTS)
                        else frozenset(
                            (
                                "config.json",
                                *(
                                    path
                                    for name in selected
                                    for path in _ARTIFACTS_BY_COMPONENT.get(name, ())
                                ),
                            )
                        )
                    )
                    report = verify_local_run(run.path, nested=False, artifact_paths=paths)
                    for code, paths in (
                        ("missing_artifact", report.missing_artifacts),
                        ("artifact_integrity", report.hash_mismatches),
                        ("invalid_path", report.invalid_paths),
                    ):
                        for path in paths:
                            if path == run.manifest.config_artifact:
                                components_for_path = ("configuration",)
                            else:
                                components_for_path = tuple(
                                    name
                                    for name in selected
                                    if path in _ARTIFACTS_BY_COMPONENT.get(name, ())
                                ) or ("manifest",)
                            for component_name in components_for_path:
                                finding(
                                    code, component_name, f"{code}: {path}", len(paths), path=path
                                )

                check("manifest", local)
                indexed_paths = {record.relative_path for record in run.manifest.artifacts}
                for component in sorted(selected):
                    for relative in _ARTIFACTS_BY_COMPONENT.get(component, ()):
                        if relative not in indexed_paths:
                            try:
                                orphan = artifact_path(run.path, relative).exists()
                            except ValueError as exc:
                                finding("invalid_path", component, str(exc), path=relative)
                            else:
                                if orphan:
                                    finding(
                                        "orphan_artifact",
                                        component,
                                        f"Unregistered artifact: {relative}",
                                        path=relative,
                                    )
                applicable = {
                    "parquet": run.parquet is not None,
                    "population": run.parquet is not None
                    and (run.manifest.initial_step == 0 or run.recovery is not None)
                    and any(a.relative_path == FOUNDERS_PATH for a in run.manifest.artifacts),
                    "snapshots": run.zarr is not None,
                    "checkpoints": run.zarr is not None,
                    "genomes": run.genomes is not None,
                    "summaries": run.summary is not None,
                    "recovery": run.recovery is not None,
                }
                for component in selected - {"manifest"}:
                    if statuses[component] is not CheckStatus.FAILED:
                        statuses[component] = (
                            CheckStatus.PASSED
                            if applicable[component]
                            else CheckStatus.NOT_APPLICABLE
                        )

                def parquet() -> None:
                    result = verify_parquet_datasets(
                        run.path, structural=structural, scientific=False
                    )
                    for code, messages in (
                        ("missing_artifact", result.missing_parts),
                        ("artifact_integrity", result.hash_mismatches),
                        ("data_schema", result.schema_mismatches),
                        ("orphan_artifact", result.orphan_parts),
                        ("index_consistency", result.consistency_errors),
                    ):
                        for message in messages:
                            finding(code, "parquet", message, len(messages))
                    if result.ok and structural:
                        check_chunks(run, scientific=scientific)
                        if scientific:
                            if run.parquet is not None and run.parquet.schema_version == 2:
                                statuses["telemetry_coverage"] = CheckStatus.PASSED
                            check_telemetry(run, finding)
                            if statuses["parquet"] is CheckStatus.FAILED:
                                statuses["telemetry_coverage"] = CheckStatus.NOT_CHECKED

                if "parquet" in selected and applicable["parquet"]:
                    check("parquet", parquet)
                if "genomes" in selected and applicable["genomes"]:
                    check(
                        "genomes",
                        lambda: [
                            finding("genome_verification", "genomes", message)
                            for message in verify_genome_store(
                                run.path, structural=structural
                            ).errors
                        ],
                    )
                for component, kind in (("snapshots", "snapshot"), ("checkpoints", "checkpoint")):
                    if component not in selected or not applicable[component]:
                        continue
                    result = verify_zarr_artifacts(run.path, structural=structural, kind=kind)
                    records = run.list_snapshots() if kind == "snapshot" else run.list_checkpoints()
                    if statuses[component] is not CheckStatus.FAILED:
                        statuses[component] = (
                            CheckStatus.PASSED if records else CheckStatus.NOT_APPLICABLE
                        )
                    for messages in (
                        result.missing_stores,
                        result.hash_mismatches,
                        result.orphan_stores,
                        result.schema_errors,
                        result.compatibility_errors,
                    ):
                        for capture_error in messages:
                            finding("capture_verification", component, capture_error, len(messages))
                    if (
                        result.ok
                        and records
                        and scientific
                        and statuses["parquet"] != CheckStatus.FAILED
                    ):

                        def captures(name: str = component) -> None:
                            def capture_issue(
                                code: str, target: str, message: str, count: int | None
                            ) -> None:
                                finding(
                                    code,
                                    target,
                                    message,
                                    count,
                                    severity=(
                                        ValidationSeverity.WARNING
                                        if code == "accumulator_saturated"
                                        else ValidationSeverity.ERROR
                                    ),
                                )

                            compared, total = check_captures(run, name, capture_issue)
                            metric_check = (
                                "checkpoint_metrics"
                                if name == "checkpoints"
                                else "snapshot_metrics"
                            )
                            if statuses[name] is CheckStatus.PASSED and compared == total:
                                statuses[metric_check] = CheckStatus.PASSED
                            if any(
                                issue.code == "capture_metrics" and issue.component == name
                                for issue in issues
                            ):
                                statuses[metric_check] = CheckStatus.FAILED
                            if compared < total:
                                finding(
                                    "capture_metric_unavailable",
                                    name,
                                    f"{total - compared} capture(s) lack a corresponding sample",
                                    total - compared,
                                    severity=ValidationSeverity.INFO,
                                )

                        check(component, captures)
                if structural:
                    if "population" in selected and applicable["population"]:
                        check("population", lambda: check_analysis(run, (FOUNDERS_PATH,)))
                    if "summaries" in selected and applicable["summaries"]:
                        check(
                            "summaries", lambda: check_analysis(run, (AGENTS_PATH, LINEAGES_PATH))
                        )
                if scientific and run.manifest.initial_step > 0 and run.recovery is None:
                    finding(
                        "trajectory_origin_unavailable",
                        "population",
                        "This run started from a checkpoint without a committed imported prefix",
                        severity=ValidationSeverity.WARNING,
                    )
                if scientific and statuses["parquet"] != CheckStatus.FAILED:
                    if (
                        "population" in selected
                        and applicable["population"]
                        and statuses["population"] is not CheckStatus.FAILED
                    ):
                        check("population", lambda: check_population(run, finding))
                    if (
                        "summaries" in selected
                        and run.summary is not None
                        and run.parquet is not None
                        and statuses["summaries"] != CheckStatus.FAILED
                        and applicable["population"]
                    ):
                        check("summaries", lambda: check_summaries(run, finding))
                        if (
                            run.manifest.final_step is not None
                            and not any(
                                c.state_step == run.manifest.final_step
                                for c in (*run.list_snapshots(), *run.list_checkpoints())
                            )
                            and (run.manifest.final_step - 1) % run.parquet.policy.metrics_interval
                        ):
                            finding(
                                "final_metric_unavailable",
                                "summaries",
                                "Final resources and energy lack a matching sample or capture",
                                severity=ValidationSeverity.INFO,
                            )
                    if "recovery" in selected and applicable["recovery"]:

                        def recovery() -> None:
                            if check_recovery(run):
                                statuses["recovery_source"] = CheckStatus.PASSED
                            else:
                                finding(
                                    "source_unavailable",
                                    "recovery_source",
                                    "Source run or checkpoint unavailable in provenance chain",
                                    severity=ValidationSeverity.WARNING,
                                )

                        check("recovery", recovery)
                if (
                    "parquet" in selected
                    and scientific
                    and run.parquet
                    and run.parquet.schema_version == 1
                ):
                    finding(
                        "legacy_coverage",
                        "parquet",
                        "Legacy index lacks committed chunk coverage",
                        severity=ValidationSeverity.WARNING,
                    )

            if not active and registry is not None:
                from sqlalchemy.exc import SQLAlchemyError

                from evolucio.persistence.postgres.reconcile import reconcile_run

                def postgres() -> None:
                    result = reconcile_run(run.path, registry)
                    if not result.ok:
                        finding("postgresql_discrepancy", "postgresql", str(result))

                try:
                    check("postgresql", postgres)
                except SQLAlchemyError as exc:
                    statuses["postgresql"] = CheckStatus.NOT_CHECKED
                    finding(
                        "postgresql_unavailable",
                        "postgresql",
                        str(exc),
                        severity=ValidationSeverity.WARNING,
                    )
            if tracking_uri is not None and registry is None:
                finding(
                    "mlflow_requires_registry",
                    "mlflow",
                    "MLflow reconciliation requires PostgreSQL binding metadata",
                    severity=ValidationSeverity.WARNING,
                )
            if not active and tracking_uri is not None and registry is not None:
                try:
                    from evolucio.tracking.mlflow.reconcile import reconcile_mlflow_run

                    result = reconcile_mlflow_run(run.path, registry, tracking_uri=tracking_uri)
                    statuses["mlflow"] = CheckStatus.PASSED
                    if not result.ok:
                        finding("mlflow_discrepancy", "mlflow", str(result))
                except (ImportError, OSError, ValueError, RuntimeError) as exc:
                    finding(
                        "mlflow_unavailable",
                        "mlflow",
                        str(exc),
                        severity=ValidationSeverity.WARNING,
                    )
    complete = data.manifest.status is RunStatus.COMPLETED if data is not None else None
    if complete is False and not allow_partial and validation_level is not ValidationLevel.METADATA:
        blocked = True
        finding(
            "partial_run",
            "manifest",
            "Validating a partial prefix requires explicit opt-in",
            severity=ValidationSeverity.WARNING,
        )
    local_passed = (
        data is not None
        and not blocked
        and not any(
            status is CheckStatus.FAILED
            for component, status in statuses.items()
            if component not in ("postgresql", "mlflow")
        )
    )
    passed = (
        local_passed
        and (registry is None or statuses["postgresql"] is CheckStatus.PASSED)
        and (tracking_uri is None or statuses["mlflow"] is CheckStatus.PASSED)
    )
    return (
        data,
        ValidationReport(
            data.manifest.run_id if data else None,
            data.manifest.status.value if data else None,
            validation_level,
            statuses,
            tuple(issues),
            tuple(
                name
                for name, status in statuses.items()
                if status in (CheckStatus.PASSED, CheckStatus.FAILED)
            ),
            tuple(name for name, status in statuses.items() if status is CheckStatus.NOT_CHECKED),
            complete,
            passed,
            local_validation_passed=local_passed,
        ),
        error,
    )


def validate_run(
    run_dir: Path,
    *,
    validation_level: ValidationLevel = ValidationLevel.METADATA,
    components: Iterable[str] = COMPONENTS,
    allow_partial: bool = False,
    registry: "PostgresRegistry | None" = None,
    tracking_uri: str | None = None,
) -> ValidationReport:
    return evaluate_run(
        run_dir,
        validation_level=validation_level,
        components=components,
        allow_partial=allow_partial,
        registry=registry,
        tracking_uri=tracking_uri,
    )[1]


@dataclass(frozen=True, slots=True)
class ExperimentValidationReport:
    experiment_id: UUID
    runs: tuple[ValidationReport, ...]
    totals: dict[str, int]


def validate_experiment(
    experiment_id: UUID,
    registry: "PostgresRegistry",
    validation_level: ValidationLevel,
    *,
    storage_root: Path = Path("runs"),
) -> ExperimentValidationReport:
    from evolucio.persistence.postgres.queries import runs_for_experiment

    totals: dict[str, int] = {
        key: 0
        for key in (
            "passed",
            "failed",
            "stopped",
            "partial",
            "invalid",
            "incompatible",
            "recovered",
            "independent",
        )
    }
    reports: list[ValidationReport] = []
    for record in runs_for_experiment(registry, experiment_id):
        run, report, _ = evaluate_run(
            registered_run_path(record.run_id, record.local_run_path, storage_root),
            validation_level=validation_level,
            allow_partial=True,
        )
        reports.append(report)
        totals["passed"] += report.validation_passed
        totals["failed"] += record.status == RunStatus.FAILED
        totals["stopped"] += record.status == RunStatus.STOPPED
        totals["partial"] += report.data_complete is False
        totals["invalid"] += not report.validation_passed
        totals["incompatible"] += any(i.code == "unsupported_schema_version" for i in report.issues)
        totals["recovered"] += run is not None and run.recovery is not None
        totals["independent"] += (
            run is not None and run.recovery is None and run.manifest.initial_step == 0
        )
    return ExperimentValidationReport(experiment_id, tuple(reports), totals)
