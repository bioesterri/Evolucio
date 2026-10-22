"""Sequential orchestration through canonical initialization and runtime modes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from evolucio.runtime.modes import run_with_mode
from evolucio.runtime.types import RunPlan

from .preflight import BatchPreflight, prepare_batch
from .types import (
    BatchDefinitionError,
    BatchExecutionError,
    BatchPlan,
    BatchResult,
    BatchRunResult,
    BatchRunSpec,
    BatchRunStatus,
    ConsumerFactory,
)

if TYPE_CHECKING:
    from evolucio.config import FrozenConfig
    from evolucio.config.compile import CompiledConfig
    from evolucio.persistence.local import LocalRunRecorder
    from evolucio.runtime.modes import ModeRunResult


def _summarize(result: ModeRunResult) -> tuple[int, int, int, int]:
    """Materialize final population identities and accumulator scalars only."""
    import jax

    runtime = result.runtime
    alive, lineages, births, deaths = jax.device_get(
        (
            runtime.state.population.alive,
            runtime.state.population.lineage_id,
            runtime.accumulator.births_total,
            runtime.accumulator.deaths_total,
        )
    )
    return (
        int(alive.sum()),
        int(births),
        int(deaths),
        len({int(lineage) for lineage in lineages[alive]}),
    )


def _execute_spec(
    spec: BatchRunSpec,
    frozen: FrozenConfig,
    compiled: CompiledConfig,
    signature: str,
    consumer_factory: ConsumerFactory | None,
    recorder: LocalRunRecorder | None = None,
) -> BatchRunResult:
    """Keep all device carries local to this run, including failure paths."""
    from evolucio.core.metrics import create_empty_metrics_accumulator
    from evolucio.runtime.initialization import initialize_simulation

    if recorder is None:
        state = initialize_simulation(compiled.core, seed=spec.seed)
        accumulator = create_empty_metrics_accumulator()
        consumer = None if consumer_factory is None else consumer_factory(spec)
        outcome = run_with_mode(
            mode=spec.mode,
            state=state,
            accumulator=accumulator,
            config=compiled.core,
            plan=RunPlan(spec.total_steps, spec.chunk_steps),
            consumer=consumer,
        )
    else:
        from evolucio.runtime.recorded_run import execute_recorded_simulation

        recorded = execute_recorded_simulation(
            recorder=recorder,
            config=compiled.core,
            plan=RunPlan(spec.total_steps, spec.chunk_steps),
            mode=spec.mode,
            consumer_factory=None if consumer_factory is None else lambda: consumer_factory(spec),
        )
        outcome = recorded.result
    alive, births, deaths, lineages = _summarize(outcome)
    return BatchRunResult(
        ordinal=spec.ordinal,
        case_id=spec.case_id,
        seed=spec.seed,
        mode=spec.mode,
        status=BatchRunStatus.COMPLETED,
        steps_requested=spec.total_steps,
        steps_executed=outcome.runtime.steps_executed,
        final_step=outcome.runtime.final_step,
        alive_count=alive,
        births_total=births,
        deaths_total=deaths,
        active_lineage_count=lineages,
        config_hash=frozen.config_hash,
        compile_signature=signature,
        run_id=None if recorder is None else recorder.run_id,
        run_dir=None if recorder is None else str(recorder.run_dir),
    )


def run_batch(
    *,
    plan: BatchPlan,
    consumer_factory: ConsumerFactory | None = None,
    output_root: Path | None = None,
    on_preflight: Callable[[BatchPreflight, tuple[BatchRunSpec, ...]], None] | None = None,
    on_result: Callable[[BatchRunResult], None] | None = None,
) -> BatchResult:
    """Validate everything first, then execute fresh independent runs in order.

    Fail-fast raises BatchExecutionError with the executed prefix. Failures do
    not expose a partial device carry: execution counts and metrics are unknown.
    Observer callbacks propagate their own exceptions; they are orchestration,
    whereas a per-run consumer failure belongs to that run.
    """
    prepared = prepare_batch(plan)
    from evolucio.config.compile import (
        ConfigCompilationError,
        compile_config,
        compile_signature_digest,
    )

    # Validate representability for every configuration before any initialization.
    try:
        compiled_configs = tuple(compile_config(frozen.config) for frozen in prepared.configs)
    except ConfigCompilationError as exc:
        raise BatchDefinitionError(str(exc)) from exc
    if on_preflight is not None:
        on_preflight(prepared.preflight, prepared.specs)
    runs: list[BatchRunResult] = []
    for spec, frozen, compiled in zip(
        prepared.specs, prepared.configs, compiled_configs, strict=True
    ):
        signature = compile_signature_digest(compiled.compile_signature)
        recorder = None
        try:
            if output_root is not None:
                from evolucio.config.versions import CONFIG_SCHEMA_VERSION
                from evolucio.host.schema import HOST_DTO_SCHEMA_DIGEST
                from evolucio.persistence.local import RunSchemas, create_local_run

                recorder = create_local_run(
                    config=frozen.config,
                    plan=RunPlan(spec.total_steps, spec.chunk_steps),
                    mode=spec.mode,
                    output_root=output_root,
                    compile_signature=signature,
                    schemas=RunSchemas(
                        CONFIG_SCHEMA_VERSION,
                        HOST_DTO_SCHEMA_DIGEST,
                        compiled.compile_signature.simulation_step_schema_digest,
                        compiled.compile_signature.metrics_schema_digest,
                    ),
                    case_id=spec.case_id,
                    batch_ordinal=spec.ordinal,
                )
            result = _execute_spec(spec, frozen, compiled, signature, consumer_factory, recorder)
        except Exception as exc:
            result = BatchRunResult(
                ordinal=spec.ordinal,
                case_id=spec.case_id,
                seed=spec.seed,
                mode=spec.mode,
                status=BatchRunStatus.FAILED,
                steps_requested=spec.total_steps,
                steps_executed=None,
                final_step=None,
                alive_count=None,
                births_total=None,
                deaths_total=None,
                active_lineage_count=None,
                config_hash=frozen.config_hash,
                compile_signature=signature,
                error_type=type(exc).__name__,
                error_message=str(exc),
                run_id=None if recorder is None else recorder.run_id,
                run_dir=None if recorder is None else str(recorder.run_dir),
            )
            runs.append(result)
            if on_result is not None:
                on_result(result)
            if not plan.continue_on_error:
                raise BatchExecutionError(tuple(runs), plan.run_count) from exc
            continue
        runs.append(result)
        if on_result is not None:
            on_result(result)
    completed = sum(run.status == BatchRunStatus.COMPLETED for run in runs)
    return BatchResult(tuple(runs), plan.run_count, completed, len(runs) - completed)


def seeds_for_case(result: BatchResult, case_id: str) -> tuple[int, ...]:
    """Preserve individual seed observations, including technical failures."""
    return tuple(run.seed for run in result.runs if run.case_id == case_id)
