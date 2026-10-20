"""Immutable host plans and scalar observations, without persistent identities."""

# pyright: reportUnnecessaryIsInstance=false

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from evolucio.config.errors import ConfigError
from evolucio.runtime.modes import RunMode
from evolucio.runtime.types import ChunkConsumer, RunPlan

from .seeds import SeedSet

BATCH_INTERFACE_VERSION = 1


class BatchDefinitionError(ConfigError):
    """Invalid batch definition; no run has started."""


@dataclass(frozen=True, slots=True)
class ExperimentCase:
    """Stable variant identity and an explicit complete configuration file."""

    case_id: str
    config_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be a nonempty string")
        if not isinstance(self.config_path, Path):
            raise TypeError("config_path must be a Path")


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """Explicit cases crossed with the same ordered seeds, sequentially."""

    cases: tuple[ExperimentCase, ...]
    seed_set: SeedSet
    total_steps: int
    chunk_steps: int
    mode: RunMode = RunMode.EXPERIMENT
    continue_on_error: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.cases, tuple):
            raise TypeError("cases must be an immutable tuple")
        if not self.cases:
            raise ValueError("at least one case is required")
        if any(not isinstance(case, ExperimentCase) for case in self.cases):
            raise TypeError("cases must contain ExperimentCase values")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("case_id must be unique within a batch")
        if not isinstance(self.seed_set, SeedSet):
            raise TypeError("seed_set must be a SeedSet")
        RunPlan(self.total_steps, self.chunk_steps)
        if not isinstance(self.mode, RunMode):
            raise TypeError("mode must be a RunMode")
        if not isinstance(self.continue_on_error, bool):
            raise TypeError("continue_on_error must be bool")
        for case in self.cases:
            if not case.config_path.is_file():
                raise BatchDefinitionError(f"not a configuration file: {case.config_path}")

    @property
    def run_count(self) -> int:
        return len(self.cases) * len(self.seed_set.seeds)


@dataclass(frozen=True, slots=True)
class BatchRunSpec:
    """One transient run in canonical case-major order; ordinal starts at 1."""

    ordinal: int
    case_id: str
    config_path: Path
    seed: int
    total_steps: int
    chunk_steps: int
    mode: RunMode

    @property
    def identity(self) -> str:
        return f"{self.case_id}/seed-{self.seed}"


class BatchRunStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BatchRunResult:
    """Scalar observations only; failed execution counts may be unavailable."""

    ordinal: int
    case_id: str
    seed: int
    mode: RunMode
    status: BatchRunStatus
    steps_requested: int
    steps_executed: int | None
    final_step: int | None
    alive_count: int | None
    births_total: int | None
    deaths_total: int | None
    active_lineage_count: int | None
    config_hash: str
    compile_signature: str
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Individual observations, never scientific conclusions or rankings."""

    runs: tuple[BatchRunResult, ...]
    requested_run_count: int
    completed_run_count: int
    failed_run_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.runs, tuple):
            raise TypeError("runs must be an immutable tuple")
        completed = sum(run.status == BatchRunStatus.COMPLETED for run in self.runs)
        failed = sum(run.status == BatchRunStatus.FAILED for run in self.runs)
        if (
            self.requested_run_count != len(self.runs)
            or self.completed_run_count != completed
            or self.failed_run_count != failed
            or completed + failed != self.requested_run_count
        ):
            raise ValueError("batch counts must match all requested individual results")


class BatchExecutionError(RuntimeError):
    """Fail-fast exception with the executed prefix; later specs never run."""

    def __init__(self, runs: tuple[BatchRunResult, ...], requested_run_count: int) -> None:
        self.runs = runs
        self.requested_run_count = requested_run_count
        failure = runs[-1]
        super().__init__(
            f"{failure.case_id}/seed-{failure.seed}: {failure.error_type}: {failure.error_message}"
        )


type ConsumerFactory = Callable[[BatchRunSpec], ChunkConsumer | None]
