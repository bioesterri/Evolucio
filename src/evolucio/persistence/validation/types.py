"""Portable validation results."""

from dataclasses import dataclass
from enum import StrEnum


class ValidationLevel(StrEnum):
    METADATA = "metadata"
    INTEGRITY = "integrity"
    STRUCTURAL = "structural"
    SCIENTIFIC = "scientific"

    def includes(self, level: "ValidationLevel") -> bool:
        levels = tuple(type(self))
        return levels.index(self) >= levels.index(level)


class ValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class CheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_CHECKED = "not_checked"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    severity: ValidationSeverity
    component: str
    message: str
    artifact_path: str | None = None
    step: int | None = None
    entity_id: int | None = None
    violation_count: int | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    run_id: str | None
    status: str | None
    validation_level: ValidationLevel
    components: dict[str, CheckStatus]
    issues: tuple[ValidationIssue, ...]
    checks_executed: tuple[str, ...]
    checks_skipped: tuple[str, ...]
    data_complete: bool | None
    validation_passed: bool
    schema_version: int = 1
    local_validation_passed: bool = False
