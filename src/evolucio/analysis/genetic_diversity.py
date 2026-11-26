"""Genetic diversity at recorded population samples and full checkpoints."""

from dataclasses import dataclass
from math import isclose, sqrt

import numpy as np
import polars as pl
from numpy.typing import NDArray

from evolucio.core.policy.schema import POLICY_PARAMETER_SPECS
from evolucio.host.dto import HostArray
from evolucio.persistence.genomes import genome_content_hash
from evolucio.persistence.loaders.run_data import RunData
from evolucio.persistence.local.artifacts import artifact_path
from evolucio.persistence.validation.types import CheckStatus
from evolucio.persistence.zarr.checkpoint import read_checkpoint_arrays

from .types import AnalysisUnavailableError, AnalysisWindow, Quality
from .windows import AnalysisRun, validated, within


def genetic_diversity_series(
    run: RunData | AnalysisRun, window: AnalysisWindow | None = None
) -> pl.LazyFrame:
    data = validated(run)
    population = pl.col("population")
    squared = pl.when(population >= 2).then(
        2.0 * population / (population - 1) * pl.col("genetic_parameter_variance")
    )
    return (
        data.run.scan_metrics()
        .filter(within("step", data.bounds(window)))
        .select(
            "run_id",
            "step",
            pl.col("alive_count").alias("population"),
            pl.col("active_lineage_count").alias("active_lineages"),
            pl.col("genetic_parameter_variance").cast(pl.Float64),
        )
        .with_columns(
            squared.alias("mean_pairwise_squared_distance_per_parameter"),
            squared.sqrt().alias("pairwise_rms_distance_per_parameter"),
            pl.lit(Quality.EXACT.value).alias("observation_quality"),
        )
        .sort("step")
    )


@dataclass(frozen=True, slots=True)
class CheckpointGeneticDiversity:
    run_id: str
    state_step: int
    population: int
    total_parameter_variance: float
    within_lineage_variance: float
    between_lineage_variance: float
    pairwise_rms_distance_per_parameter: float | None
    lineages: pl.DataFrame
    quality: Quality = Quality.EXACT


def checkpoint_genomes(data: AnalysisRun, state_step: int) -> dict[str, HostArray]:
    if not data.coverage.start_step <= state_step <= data.coverage.end_step_exclusive:
        raise AnalysisUnavailableError("checkpoint is outside the verified prefix")
    for record in data.run.list_checkpoints():
        if (
            record.state_step == state_step
            and data.report.components["checkpoints"] is CheckStatus.PASSED
        ):
            _, arrays = read_checkpoint_arrays(artifact_path(data.run.path, record.relative_path))
            return arrays
    raise AnalysisUnavailableError(f"validated checkpoint at state step {state_step} unavailable")


def _pairwise_rms(variance: float, size: int) -> float | None:
    return sqrt(2 * size / (size - 1) * variance) if size > 1 else None


def _variance_components(
    vectors: NDArray[np.float64], lineage_ids: NDArray[np.int64]
) -> tuple[float, float, float, list[tuple[int, int, float, float | None, NDArray[np.intp]]]]:
    size = len(vectors)
    if not size:
        return 0.0, 0.0, 0.0, []
    total = float(vectors.var(axis=0, ddof=0).mean())
    global_mean = vectors.mean(axis=0)
    within_lineage = between_lineage = 0.0
    order = np.argsort(lineage_ids, kind="stable")
    identifiers, counts = np.unique(lineage_ids[order], return_counts=True)
    rows: list[tuple[int, int, float, float | None, NDArray[np.intp]]] = []
    for lineage, indices in zip(identifiers, np.split(order, counts.cumsum()[:-1]), strict=True):
        group = vectors[indices]
        count = len(group)
        weight = count / size
        variance = float(group.var(axis=0, ddof=0).mean())
        within_lineage += weight * variance
        difference = group.mean(axis=0) - global_mean
        between_lineage += weight * float(np.square(difference).mean())
        rows.append((int(lineage), count, variance, _pairwise_rms(variance, count), indices))
    if not isclose(total, within_lineage + between_lineage, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("within- and between-lineage variances do not sum to total variance")
    return total, within_lineage, between_lineage, rows


def genetic_diversity_at_checkpoint(
    run: RunData | AnalysisRun,
    state_step: int,
    *,
    distinct_contents: bool = False,
) -> CheckpointGeneticDiversity:
    data = validated(run)
    arrays = checkpoint_genomes(data, state_step)
    alive = arrays["population/alive"]
    population = int(alive.sum())
    leaves = [
        arrays[f"genomes/{spec.path.replace('.', '_')}"][alive] for spec in POLICY_PARAMETER_SPECS
    ]
    vectors = np.concatenate(
        [
            leaf.reshape(population, spec.count)
            for leaf, spec in zip(leaves, POLICY_PARAMETER_SPECS, strict=True)
        ],
        axis=1,
    ).astype(np.float64)
    if not np.isfinite(vectors).all():
        raise ValueError("checkpoint includes nonfinite live genome parameters")
    total, within_lineage, between_lineage, groups = _variance_components(
        vectors, arrays["population/lineage_id"][alive]
    )
    rows = [
        (
            lineage,
            count,
            variance,
            rms,
            len({genome_content_hash(tuple(leaf[index] for leaf in leaves)) for index in indices})
            if distinct_contents
            else None,
        )
        for lineage, count, variance, rms, indices in groups
    ]
    lineages = pl.DataFrame(
        rows,
        orient="row",
        schema={
            "lineage_id": pl.Int64,
            "alive_count": pl.Int64,
            "mean_parameter_variance": pl.Float64,
            "pairwise_rms_distance_per_parameter": pl.Float64,
            "distinct_genome_contents": pl.Int64,
        },
    )
    metrics = genetic_diversity_series(data).filter(pl.col("step") == state_step - 1).collect()
    if metrics.height:
        sample = metrics.row(0, named=True)
        if (
            sample["population"] != population
            or sample["active_lineages"] != len(groups)
            or not isclose(sample["genetic_parameter_variance"], total, rel_tol=1e-5, abs_tol=1e-6)
        ):
            raise ValueError("checkpoint genetic diversity differs from the matching metric")
    return CheckpointGeneticDiversity(
        data.run.manifest.run_id,
        state_step,
        population,
        total,
        within_lineage,
        between_lineage,
        _pairwise_rms(total, population),
        lineages,
    )
