"""Host-only validation and frozen per-run configurations, without JAX."""

from dataclasses import dataclass
from pathlib import Path

from evolucio.config import ExperimentConfig, FrozenConfig, freeze_config, load_config
from evolucio.runtime.modes import RunMode

from .types import BatchPlan, BatchRunSpec


@dataclass(frozen=True, slots=True)
class CasePreflight:
    """Base document identity and canonical identities after applying each seed."""

    case_id: str
    path: Path
    config_hash: str
    run_config_hashes: tuple[tuple[int, str], ...]


@dataclass(frozen=True, slots=True)
class BatchPreflight:
    case_count: int
    seed_count: int
    run_count: int
    total_steps_per_run: int
    chunk_steps: int
    mode: RunMode
    cases: tuple[CasePreflight, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    specs: tuple[BatchRunSpec, ...]
    configs: tuple[FrozenConfig, ...]
    preflight: BatchPreflight


def expand_batch_plan(plan: BatchPlan) -> tuple[BatchRunSpec, ...]:
    """Pure case-major Cartesian expansion, preserving both explicit orders."""
    return tuple(
        BatchRunSpec(
            ordinal=case_index * len(plan.seed_set.seeds) + seed_index + 1,
            case_id=case.case_id,
            config_path=case.config_path,
            seed=seed,
            total_steps=plan.total_steps,
            chunk_steps=plan.chunk_steps,
            mode=plan.mode,
        )
        for case_index, case in enumerate(plan.cases)
        for seed_index, seed in enumerate(plan.seed_set.seeds)
    )


def prepare_batch(plan: BatchPlan) -> PreparedBatch:
    # Recheck paths at execution time, even for a previously constructed plan.
    plan.__post_init__()
    configs: list[FrozenConfig] = []
    cases: list[CasePreflight] = []
    for case in plan.cases:
        base = load_config(case.config_path)
        hashes: list[tuple[int, str]] = []
        for seed in plan.seed_set.seeds:
            data = base.model_dump(mode="python")
            data["seed"] = seed
            frozen = freeze_config(ExperimentConfig.model_validate(data))
            configs.append(frozen)
            hashes.append((seed, frozen.config_hash))
        cases.append(
            CasePreflight(
                case.case_id, case.config_path, freeze_config(base).config_hash, tuple(hashes)
            )
        )
    return PreparedBatch(
        specs=expand_batch_plan(plan),
        configs=tuple(configs),
        preflight=BatchPreflight(
            case_count=len(plan.cases),
            seed_count=len(plan.seed_set.seeds),
            run_count=plan.run_count,
            total_steps_per_run=plan.total_steps,
            chunk_steps=plan.chunk_steps,
            mode=plan.mode,
            cases=tuple(cases),
            seeds=plan.seed_set.seeds,
        ),
    )


def preflight_batch(plan: BatchPlan) -> BatchPreflight:
    """Inspect host validity without creating device arrays or initial states."""
    return prepare_batch(plan).preflight
