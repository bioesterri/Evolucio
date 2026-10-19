"""Device-independent run policies and host result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING

from .types import ChunkConsumer, RunPlan, RunRuntimeResult

if TYPE_CHECKING:
    from evolucio.config.compile import CoreConfig
    from evolucio.core.metrics import MetricsAccumulator
    from evolucio.core.state import SimulationState


class RunMode(StrEnum):
    """Operational instrumentation, never scientific model configuration."""

    SMOKE = "smoke"
    DEBUG = "debug"
    EXPERIMENT = "experiment"
    BENCHMARK = "benchmark"


@dataclass(frozen=True, slots=True)
class RunModePolicy:
    """Canonical host instrumentation switches; never passed to a kernel."""

    mode: RunMode
    per_step_diagnostics: bool = False
    initial_invariant_check: bool = False
    final_invariant_check: bool = False
    collect_telemetry: bool = True
    show_progress: bool = False
    warmup_before_timing: bool = False
    measure_runtime: bool = False


_POLICIES = MappingProxyType(
    {
        RunMode.SMOKE: RunModePolicy(
            RunMode.SMOKE, initial_invariant_check=True, final_invariant_check=True
        ),
        RunMode.DEBUG: RunModePolicy(
            RunMode.DEBUG,
            per_step_diagnostics=True,
            initial_invariant_check=True,
            final_invariant_check=True,
        ),
        RunMode.EXPERIMENT: RunModePolicy(RunMode.EXPERIMENT),
        RunMode.BENCHMARK: RunModePolicy(
            RunMode.BENCHMARK,
            collect_telemetry=False,
            warmup_before_timing=True,
            measure_runtime=True,
        ),
    }
)


def get_mode_policy(mode: RunMode) -> RunModePolicy:
    """Return the single immutable policy for a supported mode."""
    return _POLICIES[mode]


@dataclass(frozen=True, slots=True)
class InvariantDiagnostic:
    """Small host representation of one canonical invariant report."""

    ok: bool
    violation_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SmokeReport:
    """Structural start check and final viability check."""

    initial: InvariantDiagnostic
    final: InvariantDiagnostic


@dataclass(frozen=True, slots=True)
class DebugStepDiagnostic:
    """Scalar diagnostics only; no population, genomes or device arrays."""

    step: int
    initial_ok: bool
    final_ok: bool
    transition_ok: bool
    initial_violation_counts: tuple[int, ...]
    final_violation_counts: tuple[int, ...]
    transition_violation_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DebugFailure:
    """First failing phase and nonzero counts, also usable for smoke checks."""

    step: int
    phase: str
    invariant_codes: tuple[str, ...]
    violation_counts: tuple[int, ...]


class RunDiagnosticError(RuntimeError):
    """Expected invariant failure, without a state dump or repair."""

    def __init__(self, mode: RunMode, failure: DebugFailure) -> None:
        self.mode = mode
        self.failure = failure
        super().__init__(str(failure))

    def __str__(self) -> str:
        failure = self.failure
        violations = "\n".join(
            f"  {code}: {count}"
            for code, count in zip(failure.invariant_codes, failure.violation_counts, strict=True)
        )
        return (
            f"mode: {self.mode.value}\nstatus: failed\nstep: {failure.step}\n"
            f"phase: {failure.phase}\nviolations:\n{violations}"
        )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Timing of the normal host runtime, excluding initialization and warmup."""

    total_steps: int
    chunks_executed: int
    elapsed_seconds: float
    steps_per_second: float
    initial_step: int
    final_step: int
    platform: str
    device_kind: str


@dataclass(frozen=True, slots=True)
class ModeRunResult:
    """Successful finite run and optional instrumentation summaries."""

    mode: RunMode
    runtime: RunRuntimeResult
    smoke_report: SmokeReport | None = None
    latest_diagnostic: DebugStepDiagnostic | None = None
    benchmark: BenchmarkResult | None = None


def run_with_mode(
    *,
    mode: RunMode,
    state: SimulationState,
    accumulator: MetricsAccumulator,
    config: CoreConfig,
    plan: RunPlan,
    consumer: ChunkConsumer | None = None,
) -> ModeRunResult:
    """Dispatch instrumentation centrally without changing config or horizon."""
    policy = get_mode_policy(mode)
    if policy.per_step_diagnostics:
        from .diagnostics import run_debug

        return run_debug(
            state=state, accumulator=accumulator, config=config, plan=plan, consumer=consumer
        )
    if policy.measure_runtime:
        from .benchmark import run_benchmark

        return run_benchmark(state=state, accumulator=accumulator, config=config, plan=plan)
    if policy.initial_invariant_check or policy.final_invariant_check:
        from .diagnostics import run_smoke

        return run_smoke(
            state=state, accumulator=accumulator, config=config, plan=plan, consumer=consumer
        )
    from . import run_simulation

    return ModeRunResult(
        mode=mode,
        runtime=run_simulation(
            state=state, accumulator=accumulator, config=config, plan=plan, consumer=consumer
        ),
    )
