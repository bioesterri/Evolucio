"""Declarative designs resolved through the existing batch preflight."""

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StrictInt, StringConstraints, model_validator

from evolucio.analysis.types import AnalysisPolicy
from evolucio.persistence.local.artifacts import atomic_write_text, json_text
from evolucio.persistence.local.manifest import utc_now

from .batch import run_batch
from .config_diff import (
    SCIENTIFIC_FIELDS,
    ConfigDiff,
    DifferenceCategory,
    compare_configs,
)
from .preflight import BatchPreflight, PreparedBatch, prepare_batch
from .seeds import SeedSet
from .types import BatchPlan, BatchResult, BatchRunResult, BatchRunSpec, ExperimentCase


class CaseRole(StrEnum):
    BASELINE = "BASELINE"
    CONTROL = "CONTROL"
    VARIANT = "VARIANT"


class PairPolicy(StrEnum):
    STRICT = "STRICT"
    AVAILABLE_PAIRS = "AVAILABLE_PAIRS"


Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class DesignRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DesignCase(DesignRecord):
    case_id: Nonempty
    role: CaseRole
    config_path: Path
    config_hash: str | None = None
    description: str = ""


class Hypothesis(DesignRecord):
    question: Nonempty
    factor: Nonempty
    criteria: Nonempty
    expected_direction: dict[str, Literal["higher", "lower"]] = {}


class ComparisonMetrics(DesignRecord):
    primary: tuple[str, ...]
    secondary: tuple[str, ...] = ()
    exploratory: tuple[str, ...] = ()
    survival_fixed_age: StrictInt | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return self.primary + self.secondary + self.exploratory


class ExperimentDesign(DesignRecord):
    schema_version: Literal[1] = 1
    design_version: StrictInt = 1
    experiment_id: UUID
    name: Nonempty
    objective: Nonempty
    hypothesis: Hypothesis
    baseline_case_id: str
    cases: tuple[DesignCase, ...]
    seed_set: tuple[StrictInt, ...]
    total_steps: StrictInt
    chunk_steps: StrictInt
    analysis_policy: AnalysisPolicy = AnalysisPolicy()
    comparison_policy: PairPolicy = PairPolicy.STRICT
    comparison_metrics: ComparisonMetrics
    allowed_config_differences: dict[str, tuple[str, ...]]
    environment_change_step: StrictInt | None = None

    @model_validator(mode="after")
    def validate_design(self) -> Self:
        baseline_ids = [case.case_id for case in self.cases if case.role is CaseRole.BASELINE]
        if baseline_ids != [self.baseline_case_id] or len(self.cases) < 2:
            raise ValueError("expected one designated baseline and at least one contrast")
        case_ids = {case.case_id for case in self.cases}
        if len(case_ids) != len(self.cases):
            raise ValueError("case IDs must be unique")
        contrasts = case_ids - {self.baseline_case_id}
        if contrasts != self.allowed_config_differences.keys():
            raise ValueError("declare allowed scientific differences for every contrast")
        for paths in self.allowed_config_differences.values():
            if set(paths) - SCIENTIFIC_FIELDS:
                raise ValueError("unknown or non-scientific authorized parameter")
        metrics = self.comparison_metrics
        if not metrics.primary or len(metrics.names) != len(set(metrics.names)):
            raise ValueError("metrics must be unique with at least one primary metric")
        if self.hypothesis.expected_direction.keys() - set(metrics.names):
            raise ValueError("direction declared for an unplanned metric")
        if metrics.survival_fixed_age is not None and metrics.survival_fixed_age < 0:
            raise ValueError("survival_fixed_age must be nonnegative")
        if self.design_version <= 0:
            raise ValueError("design_version must be positive")
        SeedSet(self.seed_set)
        return self

    def batch_plan(self) -> BatchPlan:
        return BatchPlan(
            tuple(ExperimentCase(case.case_id, case.config_path) for case in self.cases),
            SeedSet(self.seed_set),
            self.total_steps,
            self.chunk_steps,
        )


@dataclass(frozen=True, slots=True)
class ResolvedDesign:
    design: ExperimentDesign
    batch: PreparedBatch
    differences: dict[str, tuple[ConfigDiff, ...]]
    blocking_errors: dict[str, tuple[str, ...]]


def load_design(path: Path) -> ExperimentDesign:
    design = ExperimentDesign.model_validate_json(path.read_bytes())
    return design.model_copy(
        update={
            "cases": tuple(
                case.model_copy(update={"config_path": (path.parent / case.config_path).resolve()})
                for case in design.cases
            )
        }
    )


def resolve_design(design: ExperimentDesign) -> ResolvedDesign:
    prepared = prepare_batch(design.batch_plan())
    configs = {
        spec.case_id: frozen.config
        for spec, frozen in zip(prepared.specs, prepared.configs, strict=True)
        if spec.seed == design.seed_set[0]
    }
    baseline = configs[design.baseline_case_id]
    reference = design.environment_change_step
    if reference is not None:
        phases = baseline.world.environment_schedule
        phase = next((item for item in phases if item.start_step == reference), None)
        prior = next((item for item in phases if item.end_step == reference), None)
        previous_conditions = (
            (1.0, baseline.world.environment_initial_value)
            if prior is None
            else (prior.regeneration_multiplier, prior.stress_level)
        )
        if (
            not 0 < reference < design.total_steps
            or phase is None
            or (phase.regeneration_multiplier, phase.stress_level) == previous_conditions
        ):
            raise ValueError("environment reference must match a real baseline perturbation")
    differences: dict[str, tuple[ConfigDiff, ...]] = {}
    errors: dict[str, tuple[str, ...]] = {}
    for case, preflight in zip(design.cases, prepared.preflight.cases, strict=True):
        if case.config_hash is not None and case.config_hash != preflight.config_hash:
            raise ValueError(f"{case.case_id}: configuration hash changed")
        if case.role is CaseRole.BASELINE:
            continue
        diff = compare_configs(baseline, configs[case.case_id])
        differences[case.case_id] = diff
        allowed = design.allowed_config_differences[case.case_id]
        errors[case.case_id] = tuple(
            f"Uncontrolled parameter: {item.parameter_path}"
            for item in diff
            if item.difference_category is DifferenceCategory.UNKNOWN
            or (
                item.difference_category is DifferenceCategory.SCIENTIFIC
                and item.parameter_path not in allowed
            )
        )
    return ResolvedDesign(design, prepared, differences, errors)


def plan_payload(resolved: ResolvedDesign) -> dict[str, object]:
    return {
        "design": resolved.design.model_dump(mode="json"),
        "run_count": resolved.batch.preflight.run_count,
        "config_differences": {
            case: [asdict(item) for item in changes]
            for case, changes in resolved.differences.items()
        },
        "expected_runs": [
            {"case_id": spec.case_id, "seed": spec.seed, "config_hash": frozen.config_hash}
            for spec, frozen in zip(resolved.batch.specs, resolved.batch.configs, strict=True)
        ],
        "base_config_hashes": {
            case.case_id: case.config_hash for case in resolved.batch.preflight.cases
        },
        "inclusion": "scientifically validated, completed, full planned horizon",
    }


def execute_design(resolved: ResolvedDesign, execution_dir: Path) -> BatchResult:
    if any(resolved.blocking_errors.values()):
        raise ValueError(str(resolved.blocking_errors))
    execution_dir.mkdir(parents=True, exist_ok=False)
    plan = plan_payload(resolved)
    plan["frozen_at_utc"] = utc_now()
    atomic_write_text(execution_dir / "experiment_design.json", json_text(plan))

    def check_preflight(view: BatchPreflight, specs: tuple[BatchRunSpec, ...]) -> None:
        if view != resolved.batch.preflight or specs != resolved.batch.specs:
            raise ValueError("batch preflight differs from frozen design")

    with (execution_dir / "attempts.jsonl").open("x", encoding="utf-8") as ledger:

        def record_attempt(result: BatchRunResult) -> None:
            ledger.write(json.dumps(asdict(result)) + "\n")
            ledger.flush()

        return run_batch(
            plan=resolved.design.batch_plan(),
            output_root=execution_dir / "runs",
            on_preflight=check_preflight,
            on_result=record_attempt,
        )
