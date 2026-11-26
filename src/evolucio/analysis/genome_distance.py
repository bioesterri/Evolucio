"""Distances between verified retained policies in canonical parameter order."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from math import isclose

import numpy as np
import polars as pl
from numpy.typing import NDArray

from evolucio.core.policy.schema import POLICY_PARAMETER_SPECS
from evolucio.persistence.genomes import (
    GenomeHighlightReason,
    GenomeLeaves,
    genome_content_hash,
    read_genome_leaves,
)
from evolucio.persistence.loaders.run_data import RunData

from .genealogy import genealogy_data, genealogy_nodes
from .genetic_diversity import checkpoint_genomes
from .types import AnalysisUnavailableError, Quality, positive_int
from .windows import AnalysisRun, validated


@dataclass(frozen=True, slots=True)
class GenomeDistance:
    l1_distance: float
    mean_absolute_parameter_difference: float
    l2_distance: float
    rms_parameter_difference: float
    maximum_absolute_parameter_difference: float


def flatten_genome(genome: GenomeLeaves) -> NDArray[np.float64]:
    """Validate the current policy schema and copy leaves into a Float64 vector."""
    for spec, leaf in zip(POLICY_PARAMETER_SPECS, genome, strict=True):
        if (
            leaf.shape != spec.shape
            or leaf.dtype != np.dtype(np.float32)
            or not np.isfinite(leaf).all()
        ):
            raise ValueError(f"incompatible genome leaf {spec.path}")
    return np.concatenate([leaf.ravel() for leaf in genome], dtype=np.float64)


def vector_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> GenomeDistance:
    if a.ndim != 1 or a.shape != b.shape or not a.size:
        raise ValueError("distances require two equally sized nonempty parameter vectors")
    absolute = np.abs(a - b)
    squared_mean = float(np.square(absolute).mean())
    return GenomeDistance(
        float(absolute.sum()),
        float(absolute.mean()),
        float(np.linalg.norm(absolute)),
        squared_mean**0.5,
        float(absolute.max()),
    )


def genome_distance(genome_a: GenomeLeaves, genome_b: GenomeLeaves) -> GenomeDistance:
    return vector_distance(flatten_genome(genome_a), flatten_genome(genome_b))


def retained_genome(
    run: RunData | AnalysisRun, genome_id: int, *, state_step: int | None = None
) -> GenomeLeaves:
    data = validated(run)
    for record in data.run.list_highlighted_genomes():
        if record.genome_id == genome_id and any(
            data.coverage.start_step <= step <= data.coverage.end_step_exclusive
            for step in record.reasons.values()
        ):
            try:
                return data.run.load_highlighted_genome(genome_id)
            except FileNotFoundError:
                break
    if state_step is not None:
        arrays = checkpoint_genomes(data, state_step)
        alive = arrays["population/alive"]
        matching = np.flatnonzero(alive & (arrays["population/genome_id"] == genome_id))
        if len(matching) == 1:
            slot = matching[0]
            origin = genealogy_nodes(data).filter(pl.col("genome_id") == genome_id).collect()
            if origin.height == 1 and all(
                origin[field][0] == int(arrays[f"population/{field}"][slot])
                for field in ("agent_id", "lineage_id", "generation", "birth_step")
            ):
                return tuple(
                    arrays[f"genomes/{spec.path.replace('.', '_')}"][slot]
                    for spec in POLICY_PARAMETER_SPECS
                )
    raise AnalysisUnavailableError(f"verified genome {genome_id} unavailable")


def founder_genome_divergence(run: RunData | AnalysisRun) -> pl.LazyFrame:
    data = genealogy_data(run)
    founders = {
        record.lineage_id: record
        for record in data.run.list_highlighted_genomes()
        if GenomeHighlightReason.FOUNDER in record.reasons
    }
    rows: list[dict[str, int | float]] = []
    for representative in data.run.list_highlighted_genomes():
        step = representative.reasons.get(GenomeHighlightReason.FINAL_LINEAGE_REPRESENTATIVE)
        founder = founders.get(representative.lineage_id)
        if founder is None or step is None or step != data.coverage.end_step_exclusive:
            continue
        try:
            distance = genome_distance(
                read_genome_leaves(data.run.path, founder),
                read_genome_leaves(data.run.path, representative),
            )
        except FileNotFoundError:
            continue
        rows.append(
            {
                "lineage_id": founder.lineage_id,
                "founder_agent_id": founder.agent_id,
                "founder_genome_id": founder.genome_id,
                "representative_agent_id": representative.agent_id,
                "representative_genome_id": representative.genome_id,
                "representative_generation": representative.generation,
                "state_step": step,
                **asdict(distance),
            }
        )
    schema = {
        name: pl.Int64
        for name in (
            "lineage_id",
            "founder_agent_id",
            "founder_genome_id",
            "representative_agent_id",
            "representative_genome_id",
            "representative_generation",
            "state_step",
        )
    }
    return (
        pl.DataFrame(
            rows,
            schema={**schema, **dict.fromkeys(GenomeDistance.__dataclass_fields__, pl.Float64)},
        )
        .lazy()
        .sort("lineage_id")
    )


def parent_child_genome_divergence(
    run: RunData | AnalysisRun,
    child_agent_id: int,
    *,
    parent_state_step: int | None = None,
    child_state_step: int | None = None,
) -> dict[str, object]:
    data = genealogy_data(run)
    edge = data.run.scan_genealogy().filter(pl.col("child_agent_id") == child_agent_id).collect()
    if not edge.height:
        raise ValueError(f"agent {child_agent_id} has no observed birth edge")
    row = edge.row(0, named=True)
    try:
        distance = genome_distance(
            retained_genome(data, row["parent_genome_id"], state_step=parent_state_step),
            retained_genome(data, row["child_genome_id"], state_step=child_state_step),
        )
    except AnalysisUnavailableError as exc:
        return {**row, "quality": Quality.UNAVAILABLE, "reason": str(exc)}
    return {
        **row,
        **asdict(distance),
        "quality": Quality.EXACT,
        "mutation_deltas_match": isclose(
            distance.l1_distance, row["mutation_sum_abs_delta"], rel_tol=1e-5, abs_tol=1e-6
        )
        and isclose(
            distance.maximum_absolute_parameter_difference,
            row["mutation_max_abs_delta"],
            rel_tol=1e-5,
            abs_tol=1e-6,
        ),
    }


def retained_genome_diversity(
    run: RunData | AnalysisRun,
    genome_ids: Sequence[int],
    *,
    max_genomes: int = 256,
) -> dict[str, object]:
    positive_int(max_genomes, "max_genomes")
    ids = list(dict.fromkeys(genome_ids))
    if len(ids) > max_genomes:
        raise ValueError("explicit genome selection exceeds max_genomes")
    data = validated(run)
    contents: list[GenomeLeaves] = []
    unavailable_ids: list[int] = []
    for genome_id in ids:
        try:
            contents.append(retained_genome(data, genome_id))
        except AnalysisUnavailableError:
            unavailable_ids.append(genome_id)
    vectors = [flatten_genome(genome) for genome in contents]
    totals = np.zeros(2)
    pairs = 0
    for a, b in combinations(vectors, 2):
        delta = a - b
        totals += (np.abs(delta).sum(), np.linalg.norm(delta))
        pairs += 1
    return {
        "run_id": data.run.manifest.run_id,
        "genome_ids": ids,
        "scope": "explicit_selection_of_highlighted_genomes",
        "quality": Quality.EXACT if contents else Quality.UNAVAILABLE,
        "unavailable_genome_ids": unavailable_ids,
        "genomes_available": len(contents),
        "distinct_genome_contents": len({genome_content_hash(genome) for genome in contents}),
        "pairs_compared": pairs,
        "mean_pairwise_l1_distance": float(totals[0] / pairs) if pairs else None,
        "mean_pairwise_l2_distance": float(totals[1] / pairs) if pairs else None,
    }
