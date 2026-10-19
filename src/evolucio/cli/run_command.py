"""Host orchestration and bounded retention of run summary metrics."""

from dataclasses import dataclass
from pathlib import Path

from evolucio.config import FrozenConfig
from evolucio.config.errors import ConfigError
from evolucio.host import ChunkTelemetryDTO, StepMetricsDTO
from evolucio.runtime import RunMode, RunPlan, get_mode_policy

from .config_commands import load_frozen
from .formatting import text_fields


@dataclass(slots=True)
class CliRunSummaryConsumer:
    """Retain only the last scalar record and cumulative event counts."""

    latest_metrics: StepMetricsDTO | None = None
    births_total: int = 0
    deaths_total: int = 0

    def __call__(self, batch: ChunkTelemetryDTO) -> None:
        for metrics in batch.metrics:
            self.latest_metrics = metrics
            self.births_total += metrics.birth_count
            self.deaths_total += metrics.death_count


def run_config(
    path: Path,
    *,
    steps: int,
    chunk_steps: int | None,
    quiet: bool,
    dry_run: bool,
    mode: RunMode = RunMode.EXPERIMENT,
) -> str:
    """Prepare a plan or execute through PR-30 and the PR-31 DTO adapter."""
    policy = get_mode_policy(mode)
    frozen = load_frozen(path)
    plan = RunPlan(
        total_steps=steps,
        chunk_steps=frozen.config.runtime.chunk_size if chunk_steps is None else chunk_steps,
    )
    if dry_run:
        full_chunks, remainder = divmod(plan.total_steps, plan.chunk_steps)
        return text_fields(
            {
                "mode": mode.value,
                "diagnostics_policy": {
                    "per_step": policy.per_step_diagnostics,
                    "initial": policy.initial_invariant_check,
                    "final": policy.final_invariant_check,
                },
                "benchmark_timing": policy.measure_runtime,
                "warmup_before_timing": policy.warmup_before_timing,
                "config_path": str(path),
                "config_hash": frozen.config_hash,
                "seed": frozen.config.seed,
                "total_steps": plan.total_steps,
                "chunk_steps": plan.chunk_steps,
                "chunks": full_chunks + int(remainder > 0),
                "full_chunks": full_chunks,
                "remainder": remainder,
            }
        )
    return _execute(frozen, plan=plan, quiet=quiet, mode=mode)


def _execute(frozen: FrozenConfig, *, plan: RunPlan, quiet: bool, mode: RunMode) -> str:
    """Import device-facing APIs only after validation and dry-run planning."""
    from evolucio.config.compile import ConfigCompilationError, compile_config
    from evolucio.core.metrics import create_empty_metrics_accumulator
    from evolucio.runtime import DTOChunkConsumer, run_with_mode
    from evolucio.runtime.initialization import InitializationError, initialize_simulation

    try:
        compiled = compile_config(frozen.config)
        state = initialize_simulation(compiled.core, seed=frozen.config.seed)
    except (ConfigCompilationError, InitializationError) as exc:
        raise ConfigError(str(exc)) from exc
    consumer = CliRunSummaryConsumer()
    result = run_with_mode(
        mode=mode,
        state=state,
        accumulator=create_empty_metrics_accumulator(),
        config=compiled.core,
        plan=plan,
        consumer=DTOChunkConsumer(consumer),
    )
    if result.benchmark is not None:
        benchmark = result.benchmark
        return text_fields(
            {
                "mode": result.mode.value,
                "status": "completed",
                "warmup": "completed",
                "steps": benchmark.total_steps,
                "chunks_executed": benchmark.chunks_executed,
                "initial_step": benchmark.initial_step,
                "final_step": benchmark.final_step,
                "elapsed_seconds": benchmark.elapsed_seconds,
                "steps_per_second": benchmark.steps_per_second,
                "platform": benchmark.platform,
                "device_kind": benchmark.device_kind,
            }
        )
    latest = consumer.latest_metrics
    if latest is None:
        raise RuntimeError("runtime completed without metrics")
    summary: dict[str, object] = {
        "status": "completed",
        "steps_executed": result.runtime.steps_executed,
        "final_step": result.runtime.final_step,
        "alive_count": latest.alive_count,
        "births_total": consumer.births_total,
        "deaths_total": consumer.deaths_total,
        "active_lineage_count": latest.active_lineage_count,
        "final_total_energy": latest.total_energy_alive,
        "final_total_resources": latest.total_resources,
        "population_extinct": latest.alive_count == 0,
    }
    if result.smoke_report is not None:
        summary.update(mode=result.mode.value, initial_invariants="ok", final_invariants="ok")
    if result.latest_diagnostic is not None:
        summary["mode"] = result.mode.value
    return "" if quiet else text_fields(summary)
