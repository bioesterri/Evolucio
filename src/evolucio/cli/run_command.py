"""Host orchestration and bounded retention of run summary metrics."""

from dataclasses import dataclass
from pathlib import Path

from evolucio.config import FrozenConfig
from evolucio.config.errors import ConfigError
from evolucio.host import ChunkTelemetryDTO, StepMetricsDTO
from evolucio.runtime import RunPlan

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
    path: Path, *, steps: int, chunk_steps: int | None, quiet: bool, dry_run: bool
) -> str:
    """Prepare a plan or execute through PR-30 and the PR-31 DTO adapter."""
    frozen = load_frozen(path)
    plan = RunPlan(
        total_steps=steps,
        chunk_steps=frozen.config.runtime.chunk_size if chunk_steps is None else chunk_steps,
    )
    if dry_run:
        full_chunks, remainder = divmod(plan.total_steps, plan.chunk_steps)
        return text_fields(
            {
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
    return _execute(frozen, plan=plan, quiet=quiet)


def _execute(frozen: FrozenConfig, *, plan: RunPlan, quiet: bool) -> str:
    """Import device-facing APIs only after validation and dry-run planning."""
    from evolucio.config.compile import ConfigCompilationError, compile_config
    from evolucio.core.metrics import create_empty_metrics_accumulator
    from evolucio.runtime import DTOChunkConsumer, run_simulation
    from evolucio.runtime.initialization import InitializationError, initialize_simulation

    try:
        compiled = compile_config(frozen.config)
        state = initialize_simulation(compiled.core, seed=frozen.config.seed)
    except (ConfigCompilationError, InitializationError) as exc:
        raise ConfigError(str(exc)) from exc
    consumer = CliRunSummaryConsumer()
    result = run_simulation(
        state=state,
        accumulator=create_empty_metrics_accumulator(),
        config=compiled.core,
        plan=plan,
        consumer=DTOChunkConsumer(consumer),
    )
    latest = consumer.latest_metrics
    if latest is None:
        raise RuntimeError("runtime completed without metrics")
    summary: dict[str, object] = {
        "status": "completed",
        "steps_executed": result.steps_executed,
        "final_step": result.final_step,
        "alive_count": latest.alive_count,
        "births_total": consumer.births_total,
        "deaths_total": consumer.deaths_total,
        "active_lineage_count": latest.active_lineage_count,
        "final_total_energy": latest.total_energy_alive,
        "final_total_resources": latest.total_resources,
        "population_extinct": latest.alive_count == 0,
    }
    return "" if quiet else text_fields(summary)
