"""Select at most one verified trajectory in each planned seed block."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from evolucio.analysis.types import Coverage
from evolucio.analysis.windows import AnalysisRun, validated
from evolucio.experiments.config_diff import configuration_group_hash
from evolucio.experiments.design import ExperimentDesign, ResolvedDesign, plan_payload
from evolucio.persistence.loaders.run_data import open_run, registered_run_path
from evolucio.persistence.local.exclusion import read_exclusion
from evolucio.persistence.local.manifest import RunStatus
from evolucio.persistence.validation.types import CheckStatus

if TYPE_CHECKING:
    from evolucio.persistence.postgres.repository import PostgresRegistry

type Cell = tuple[str, int]


@dataclass(frozen=True, slots=True)
class RunCandidate:
    path: Path | None
    case_id: str | None = None
    seed: int | None = None
    run_id: str | None = None
    config_hash: str | None = None
    status: str = "unknown"
    excluded_reason: str | None = None
    error: str | None = None


type RunProvider = Callable[[ExperimentDesign], Iterable[RunCandidate]]


@dataclass(frozen=True, slots=True)
class LocalRunProvider:
    root: Path
    planned_execution: bool = False

    def __call__(self, design: ExperimentDesign) -> Iterable[RunCandidate]:
        root = self.root / "runs" if self.planned_execution else self.root
        seen: set[Path] = set()
        if self.planned_execution and (self.root / "attempts.jsonl").is_file():
            for line in (
                (self.root / "attempts.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
            ):
                if not line.endswith("\n"):
                    break  # An interrupted append does not describe an attempt.
                record = json.loads(line)
                path = root / str(UUID(record["run_id"])) if record["run_id"] else None
                if path is not None:
                    seen.add(path)
                yield RunCandidate(
                    path,
                    record["case_id"],
                    record["seed"],
                    record["run_id"],
                    record["config_hash"],
                    record["status"],
                    error=record["error_message"],
                )
        if root.exists():
            for manifest in sorted(root.glob("*/manifest.json")):
                if manifest.parent not in seen:
                    yield RunCandidate(manifest.parent)

    def matches_frozen_design(self, resolved: ResolvedDesign) -> bool:
        if not self.planned_execution:
            return False
        saved = json.loads((self.root / "experiment_design.json").read_text(encoding="utf-8"))
        frozen_at = saved.pop("frozen_at_utc", None)
        frozen_design = ExperimentDesign.model_validate(saved.pop("design"))
        current = plan_payload(resolved)
        current.pop("design")
        semantic_fields = {"cases": {"__all__": {"config_path"}}}
        return (
            isinstance(frozen_at, str)
            and saved == current
            and frozen_design.model_dump(exclude=semantic_fields)
            == resolved.design.model_dump(exclude=semantic_fields)
        )


@dataclass(frozen=True, slots=True)
class RegisteredRunProvider:
    registry: PostgresRegistry
    storage_root: Path

    def __call__(self, design: ExperimentDesign) -> Iterable[RunCandidate]:
        for record in self.registry.list_runs(experiment_id=design.experiment_id):
            exclusion = self.registry.get_analysis_exclusion(record.run_id)
            yield RunCandidate(
                registered_run_path(record.run_id, record.local_run_path, self.storage_root),
                record.case_id,
                record.seed,
                str(record.run_id),
                record.config_hash,
                record.status,
                None if exclusion is None else exclusion.reason,
            )


@dataclass(frozen=True, slots=True)
class Attempt:
    case_id: str | None
    seed: int | None
    run_id: str | None
    root_run_id: str | None
    config_hash: str | None
    status: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SelectionReport:
    selected: dict[Cell, AnalysisRun]
    attempts: tuple[Attempt, ...]
    cells: dict[Cell, str]
    completed_cells: frozenset[Cell]


def select_comparison_runs(
    resolved: ResolvedDesign,
    run_provider: RunProvider,
    *,
    choices: Mapping[Cell, str] | None = None,
) -> SelectionReport:
    choices = {} if choices is None else choices
    expected = {
        (spec.case_id, spec.seed): frozen
        for spec, frozen in zip(resolved.batch.specs, resolved.batch.configs, strict=True)
    }
    if choices.keys() - expected.keys():
        raise ValueError("explicit selection refers to an unplanned cell")
    available: dict[Cell, dict[str, AnalysisRun]] = defaultdict(dict)
    attempts: list[Attempt] = []
    states: dict[Cell, set[str]] = defaultdict(set)
    completed: set[Cell] = set()
    for candidate in run_provider(resolved.design):
        case_id, seed = candidate.case_id, candidate.seed
        run_id, config_hash = candidate.run_id, candidate.config_hash
        root_run_id = None
        status, reason = candidate.status, candidate.error
        excluded = candidate.excluded_reason
        try:
            if candidate.path is not None:
                run = open_run(candidate.path, allow_partial=True)
                manifest = run.manifest
                for supplied, found in (
                    (case_id, manifest.case_id),
                    (seed, manifest.seed),
                    (run_id, manifest.run_id),
                    (config_hash, manifest.config_hash),
                ):
                    if supplied is not None and supplied != found:
                        raise ValueError("provider identity disagrees with persisted run")
                case_id, seed = manifest.case_id, manifest.seed
                run_id, config_hash = manifest.run_id, manifest.config_hash
                root_run_id = run.recovery.root_run_id if run.recovery else run_id
                cell = (case_id, seed)
                status = manifest.status.value
                exclusion = read_exclusion(run.path, manifest.run_id)
                if exclusion is not None:
                    excluded = exclusion.reason
                if case_id is None or cell not in expected:
                    raise ValueError("case or seed outside the planned matrix")
                if manifest.status is RunStatus.COMPLETED:
                    completed.add((case_id, seed))
                if excluded is not None:
                    status, reason = "discarded", excluded
                elif manifest.error is not None:
                    reason = manifest.error.message
                elif manifest.stop is not None:
                    reason = manifest.stop.reason
                elif manifest.status is RunStatus.COMPLETED:
                    frozen = expected[(case_id, seed)]
                    if config_hash != frozen.config_hash and (
                        configuration_group_hash(run.config)
                        != configuration_group_hash(frozen.config)
                    ):
                        raise ValueError("configuration differs from the planned configuration")
                    analysis = validated(run)
                    if not analysis.population_available:
                        raise ValueError("validated founder census and population data required")
                    if (
                        run.recovery is not None
                        and analysis.report.components["recovery_source"] is not CheckStatus.PASSED
                    ):
                        raise ValueError("recovery source chain could not be verified")
                    if (
                        manifest.final_step != resolved.design.total_steps
                        or analysis.coverage != Coverage(0, resolved.design.total_steps)
                    ):
                        raise ValueError("committed trajectory does not cover the planned horizon")
                    available[(case_id, seed)][manifest.run_id] = analysis
            elif status == "completed":
                raise ValueError("completed attempt has no source run")
        except (OSError, ValueError) as exc:
            status, reason = "invalid", str(exc)
        attempts.append(Attempt(case_id, seed, run_id, root_run_id, config_hash, status, reason))
        if case_id is not None and seed is not None:
            states[(case_id, seed)].add(status)

    selected: dict[Cell, AnalysisRun] = {}
    cells: dict[Cell, str] = {}
    for cell in expected:
        candidates = available[cell]
        if cell in choices:
            run_id = choices[cell]
            if run_id not in candidates:
                raise ValueError(f"{cell}: explicit selection is not an eligible candidate")
            selected[cell] = candidates[run_id]
        elif len(candidates) == 1:
            selected[cell] = next(iter(candidates.values()))
        cells[cell] = (
            "completed"
            if cell in selected
            else "ambiguous"
            if candidates
            else next(
                (
                    state
                    for state in (
                        "running",
                        "created",
                        "invalid",
                        "discarded",
                        "failed",
                        "stopped",
                        "not_started",
                    )
                    if state in states[cell]
                ),
                "unknown",
            )
        )

    roots = {
        cell: item.run.recovery.root_run_id if item.run.recovery else item.run.manifest.run_id
        for cell, item in selected.items()
    }
    counts = Counter(roots.values())
    for cell, root in roots.items():
        if counts[root] > 1:
            del selected[cell]
            cells[cell] = "invalid"
    return SelectionReport(selected, tuple(attempts), cells, frozenset(completed))
