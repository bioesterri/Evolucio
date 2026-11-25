"""Evaluate analysis and restore compatibility separately from readability."""

from dataclasses import dataclass
from pathlib import Path

from evolucio.persistence.loaders.run_data import RunData, open_run


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    readable: bool
    analysis_compatible: bool | None
    restore_compatible: bool | None
    incompatibilities: tuple[str, ...]
    differences: tuple[str, ...]
    warnings: tuple[str, ...]


def check_read_compatibility(run_dir: Path) -> CompatibilityReport:
    try:
        open_run(run_dir)
    except (OSError, ValueError) as exc:
        return CompatibilityReport(False, None, None, (str(exc),), (), ())
    return CompatibilityReport(True, None, None, (), (), ())


def compare_runs(left: RunData, right: RunData) -> CompatibilityReport:
    """Assess comparison of sampled step metrics over a shared trajectory."""
    differences: list[str] = []
    incompatible: list[str] = []
    warnings: list[str] = []
    unknown = False
    for label, a, b in (
        (
            "simulation_step",
            left.manifest.schemas.simulation_step,
            right.manifest.schemas.simulation_step,
        ),
        ("metric_definitions", left.manifest.schemas.metrics, right.manifest.schemas.metrics),
        (
            "observation_schema",
            left.config.observations.schema_version,
            right.config.observations.schema_version,
        ),
        ("policy_schema", left.config.policy.schema_version, right.config.policy.schema_version),
        ("genome_schema", left.config.genome.schema_version, right.config.genome.schema_version),
    ):
        if a is None or b is None:
            warnings.append(f"{label} is unavailable")
            unknown = True
        elif a != b:
            incompatible.append(label)
    if left.manifest.code_version != right.manifest.code_version:
        differences.append("code_version")
        warnings.append("Source versions differ; review model changes before comparing experiments")
        unknown = True
    if left.manifest.config_hash != right.manifest.config_hash:
        differences.append("config_hash")
    if left.recovery or right.recovery:
        warnings.append("Recovered trajectories share observations with their source runs")
    if left.parquet is None or right.parquet is None:
        unknown = True
        warnings.append("Sampled telemetry is unavailable")
    else:
        if left.parquet.policy.metrics_interval != right.parquet.policy.metrics_interval:
            incompatible.append("metrics_interval")
        a_steps = (
            left.parquet.datasets["metrics"].min_step,
            left.parquet.datasets["metrics"].max_step,
        )
        b_steps = (
            right.parquet.datasets["metrics"].min_step,
            right.parquet.datasets["metrics"].max_step,
        )
        if a_steps != b_steps:
            differences.append("temporal_coverage")
        if None in (*a_steps, *b_steps):
            unknown = True
            warnings.append("At least one run has no metric samples")
        elif (a_steps[1] is not None and b_steps[0] is not None and a_steps[1] < b_steps[0]) or (
            b_steps[1] is not None and a_steps[0] is not None and b_steps[1] < a_steps[0]
        ):
            incompatible.append("no_shared_sampled_steps")
        if not left.parquet.complete or not right.parquet.complete:
            warnings.append("At least one run has partial telemetry")
    return CompatibilityReport(
        True,
        False if incompatible else None if unknown else True,
        None,
        tuple(incompatible),
        tuple(differences),
        tuple(warnings),
    )


def check_restore_compatibility(run: RunData, step: int) -> CompatibilityReport:
    """Explicitly ask the checkpoint loader to rebuild its JAX state."""
    from evolucio.config.compile import build_compile_signature, compile_signature_digest
    from evolucio.persistence.zarr.checkpoint import load_checkpoint

    for record in run.list_checkpoints():
        if record.state_step == step:
            try:
                load_checkpoint(
                    checkpoint_path=run.path / record.relative_path,
                    expected_config_hash=run.manifest.config_hash,
                    expected_compile_signature=compile_signature_digest(
                        build_compile_signature(run.config)
                    ),
                )
            except (OSError, ValueError, RuntimeError) as exc:
                return CompatibilityReport(True, None, False, (str(exc),), (), ())
            return CompatibilityReport(True, None, True, (), (), ())
    return CompatibilityReport(True, None, None, (), (), (f"No checkpoint at state step {step}",))
