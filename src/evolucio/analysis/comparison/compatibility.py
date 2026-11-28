"""Experimental eligibility across configurations and independent seed blocks."""

from dataclasses import dataclass
from functools import cache

from evolucio.analysis.windows import AnalysisRun
from evolucio.experiments.config_diff import (
    CAPTURE_FIELDS,
    ConfigDiff,
    DifferenceCategory,
    compare_configs,
)
from evolucio.experiments.design import PairPolicy, ResolvedDesign
from evolucio.persistence.loaders.compatibility import compare_runs

from .selection import SelectionReport


@dataclass(frozen=True, slots=True)
class ComparabilityReport:
    experiment_id: str
    baseline_case_id: str
    variant_case_id: str
    expected_seeds: tuple[int, ...]
    baseline_available_seeds: tuple[int, ...]
    variant_available_seeds: tuple[int, ...]
    paired_seeds: tuple[int, ...]
    missing_seeds: tuple[int, ...]
    config_differences: tuple[ConfigDiff, ...]
    schema_differences: tuple[str, ...]
    coverage_differences: tuple[str, ...]
    blocking_errors: tuple[str, ...]
    warnings: tuple[str, ...]
    comparison_eligible: bool


def compare_compatibility(
    resolved: ResolvedDesign,
    selection: SelectionReport,
    case_id: str,
    policy: PairPolicy,
) -> ComparabilityReport:
    design = resolved.design
    baseline_id = design.baseline_case_id
    seeds = design.seed_set
    selected = selection.selected
    baseline = tuple(seed for seed in seeds if (baseline_id, seed) in selected)
    variant = tuple(seed for seed in seeds if (case_id, seed) in selected)
    candidate_pairs = tuple(seed for seed in seeds if seed in baseline and seed in variant)
    pairs: list[int] = []
    schemas: set[str] = set()
    coverage: set[str] = set()
    warnings: set[str] = set()
    blocking = list(resolved.blocking_errors[case_id])
    config_differences = list(resolved.differences[case_id])
    coverage.update(
        f"Declared capture setting differs: {item.parameter_path}"
        for item in config_differences
        if item.parameter_path in CAPTURE_FIELDS
    )

    @cache
    def signature_matches(run_id: str) -> bool:
        from evolucio.config.compile import build_compile_signature, compile_signature_digest

        run = runs_by_id[run_id].run
        return run.manifest.compile_signature == compile_signature_digest(
            build_compile_signature(run.config)
        )

    runs_by_id = {item.run.manifest.run_id: item for item in selected.values()}

    def compatible(left: AnalysisRun, right: AnalysisRun, seed: int) -> bool:
        outcome = compare_runs(left.run, right.run)
        warnings.update(outcome.warnings)
        a, b = left.run.manifest, right.run.manifest
        differences = set(outcome.incompatibilities) - {
            "metrics_interval",
            "no_shared_sampled_steps",
        }
        if a.code_version != b.code_version or a.code_version.git_commit is None:
            differences.add("source provenance")
        elif a.code_version.git_dirty is not False:
            if policy is PairPolicy.STRICT:
                differences.add("unverified working-tree provenance")
            else:
                warnings.add("Source commit has unverified working-tree changes")
        if a.environment != b.environment or a.environment.backend is None:
            differences.add("execution environment or backend")
        if any(
            item.run.config.runtime.backend == "cpu"
            and item.run.manifest.environment.backend != "cpu"
            for item in (left, right)
        ):
            differences.add("backend differs from the declared CPU backend")
        if a.schemas != b.schemas or a.schemas.metrics is None or a.schemas.simulation_step is None:
            differences.add("manifest schemas")
        if (
            left.run.parquet
            and right.run.parquet
            and {name: index.schema_version for name, index in left.run.parquet.datasets.items()}
            != {name: index.schema_version for name, index in right.run.parquet.datasets.items()}
        ):
            differences.add("telemetry schemas")
        if not signature_matches(a.run_id) or not signature_matches(b.run_id):
            differences.add("unverified compile signature (includes RNG and mutation schema)")
        schemas.update(f"seed {seed}: {item}" for item in differences)
        if left.metrics_interval != right.metrics_interval:
            coverage.add(f"seed {seed}: different telemetry sampling intervals")
        if "no_shared_sampled_steps" in outcome.incompatibilities:
            coverage.add(f"seed {seed}: no shared sampled steps")
        if "temporal_coverage" in outcome.differences:
            coverage.add(f"seed {seed}: different sampled step ranges")
        return not differences

    anchor: AnalysisRun | None = None
    for seed in candidate_pairs:
        left = selected[(baseline_id, seed)]
        right = selected[(case_id, seed)]
        for difference in compare_configs(left.run.config, right.run.config):
            if difference not in config_differences:
                config_differences.append(difference)
        if left.run.manifest.chunk_steps != right.run.manifest.chunk_steps:
            difference = ConfigDiff(
                "manifest.chunk_steps",
                left.run.manifest.chunk_steps,
                right.run.manifest.chunk_steps,
                DifferenceCategory.OPERATIONAL,
            )
            if difference not in config_differences:
                config_differences.append(difference)
        if compatible(left, right, seed):
            if anchor is None:
                anchor = left
            if anchor is left or compatible(anchor, left, seed):
                pairs.append(seed)
            elif "Execution provenance differs across seed blocks" not in blocking:
                blocking.append("Execution provenance differs across seed blocks")

    missing = tuple(seed for seed in seeds if seed not in pairs)
    if not pairs:
        blocking.append("No verified compatible pairs")
    if policy is PairPolicy.STRICT and missing:
        blocking.append("STRICT requires every planned seed pair")
    if any(
        selection.cells[(case, seed)] == "ambiguous"
        for case in (baseline_id, case_id)
        for seed in seeds
    ):
        blocking.append("Ambiguous completed runs require explicit selection")
    if len(pairs) < 2:
        warnings.add("Between-seed variability requires at least two pairs")
    if missing:
        warnings.add("Incomplete coverage can bias the observed effect")
    if policy is PairPolicy.AVAILABLE_PAIRS:
        warnings.add("Available-pairs comparison is exploratory")
    return ComparabilityReport(
        str(design.experiment_id),
        baseline_id,
        case_id,
        seeds,
        baseline,
        variant,
        tuple(pairs),
        missing,
        tuple(config_differences),
        tuple(sorted(schemas)),
        tuple(sorted(coverage)),
        tuple(blocking),
        tuple(sorted(warnings)),
        not blocking,
    )
