"""Reference values for genealogy, diversity and genetic comparisons."""

from math import log, sqrt
from types import SimpleNamespace
from typing import cast

import numpy as np
import polars as pl
import pytest

from evolucio.analysis.genealogy import get_ancestors, get_descendants
from evolucio.analysis.genetic_diversity import _pairwise_rms, _variance_components
from evolucio.analysis.genome_distance import (
    flatten_genome,
    parent_child_genome_divergence,
    retained_genome_diversity,
    vector_distance,
)
from evolucio.analysis.lineage_diversity import _diversity, lineage_proportions_at_step
from evolucio.analysis.lineage_dynamics import lineage_population_series, survival_curve_by_lineage
from evolucio.analysis.mutation_analysis import (
    ancestral_mutation_totals,
    mutation_summary_by_generation,
    mutation_summary_by_lineage,
    mutation_summary_by_window,
)
from evolucio.analysis.offspring import (
    generation_summary,
    offspring_distribution,
    reproductive_success_by_agent,
)
from evolucio.analysis.summaries import FOUNDERS_PATH
from evolucio.analysis.types import Coverage, Quality
from evolucio.analysis.windows import AnalysisRun
from evolucio.core.policy.schema import POLICY_PARAMETER_COUNT, POLICY_PARAMETER_SPECS
from evolucio.persistence.genomes import GenomeHighlightReason, GenomeLeaves
from evolucio.persistence.loaders.run_data import RunData
from evolucio.persistence.local.manifest import RunStatus
from evolucio.persistence.validation.types import CheckStatus, ValidationReport


def _forest() -> AnalysisRun:
    founders = pl.LazyFrame(
        {
            "agent_id": [0],
            "lineage_id": [0],
            "genome_id": [10],
            "generation": [0],
            "birth_step": [0],
        }
    )
    births = pl.LazyFrame(
        {
            "parent_agent_id": [0, 0, 1, 3],
            "child_agent_id": [1, 2, 3, 4],
            "parent_genome_id": [10, 10, 11, 13],
            "child_genome_id": [11, 12, 13, 14],
            "lineage_id": [0] * 4,
            "generation": [1, 1, 2, 3],
            "birth_step": [0, 1, 2, 4],
            "mutation_effective_count": [1, 0, 1, 0],
            "mutation_selected_count": [1, 0, 1, 0],
            "mutation_selected_weight_count": [1, 0, 1, 0],
            "mutation_selected_bias_count": [0, 0, 0, 0],
            "mutation_sum_abs_delta": [1.0, 0.0, 1.0, 0.0],
            "mutation_max_abs_delta": [1.0, 0.0, 1.0, 0.0],
        }
    )
    deaths = pl.LazyFrame({"agent_id": [2], "death_step": [2], "age": [1], "lineage_id": [0]})
    genome: GenomeLeaves = tuple(
        np.zeros(spec.shape, dtype=np.float32) for spec in POLICY_PARAMETER_SPECS
    )
    source = SimpleNamespace(
        manifest=SimpleNamespace(
            run_id="test",
            final_step=6,
            status=RunStatus.STOPPED,
            artifacts=[SimpleNamespace(relative_path=FOUNDERS_PATH)],
        ),
        scan_founders=lambda: founders,
        scan_births=lambda: births,
        scan_deaths=lambda: deaths,
        scan_genealogy=lambda: births,
        list_highlighted_genomes=lambda: [
            SimpleNamespace(genome_id=genome_id, reasons={GenomeHighlightReason.FOUNDER: 0})
            for genome_id in (10, 14, 15)
        ],
        load_highlighted_genome=lambda _: genome,
    )
    report = SimpleNamespace(components={"population": CheckStatus.PASSED})
    return AnalysisRun(cast(RunData, source), cast(ValidationReport, report), Coverage(0, 6), 10)


def test_ancestry_descendants_and_offspring_censoring() -> None:
    run = _forest()
    result = get_descendants(run, 0)
    assert result.direct_offspring_count == 2
    assert result.total_descendant_count == 4
    assert result.descendants["generation"].max() == 3
    assert not result.partial
    assert get_ancestors(run, 4)["agent_id"].to_list() == [3, 1, 0]
    assert get_ancestors(run, 0).is_empty()
    assert get_descendants(run, 0, max_depth=1).descendants["descendant_agent_id"].to_list() == [
        1,
        2,
    ]
    assert get_descendants(run, 0, max_depth=1).partial
    assert get_descendants(run, 0, max_nodes=3).partial
    assert not get_descendants(run, 0, max_nodes=4).partial
    with pytest.raises(ValueError, match="unknown agent_id"):
        get_ancestors(run, 99)
    distribution = offspring_distribution(run)
    assert distribution.summary["agents_total"] == 5
    assert distribution.summary["agents_without_offspring_observed"] == 2
    assert distribution.summary["mean_direct_offspring"] == 0.8
    assert distribution.summary["median_direct_offspring"] == 1
    assert distribution.histogram.collect()["agent_count"].to_list() == [2, 2, 1]
    observations = reproductive_success_by_agent(run).collect()
    assert observations.filter(pl.col("died_without_offspring"))["agent_id"].to_list() == [2]
    assert observations.filter(pl.col("reproduction_right_censored"))["agent_id"].to_list() == [4]
    assert observations["first_reproduction_step"].to_list() == [0, 2, None, 4, None]
    assert observations["time_to_first_reproduction"].to_list() == [0, 1, None, 1, None]
    common = generation_summary(run, follow_up_steps=3).collect()
    assert common["generation"].to_list() == [0, 1, 2]
    assert common["agents_dead"].to_list() == [0, 1, 0]
    one_step = generation_summary(run, follow_up_steps=1).collect()
    assert one_step["mean_direct_offspring"].to_list() == [1.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("counts", "entropy", "concentration", "evenness"),
    [
        ([25] * 4, log(4), 0.25, 1.0),
        ([90, 10], -0.9 * log(0.9) - 0.1 * log(0.1), 0.82, 0.4689955935892812),
        ([100], 0.0, 1.0, None),
        ([], None, None, None),
    ],
)
def test_lineage_diversity_reference(
    counts: list[int], entropy: float | None, concentration: float | None, evenness: float | None
) -> None:
    events = pl.DataFrame(
        {
            "step": [0] * len(counts),
            "lineage_id": list(range(len(counts))),
            "population_after": counts,
        },
        schema=dict.fromkeys(("step", "lineage_id", "population_after"), pl.Int64),
    )
    result = _diversity(events, 0, 1).row(0, named=True)
    assert result["active_lineages"] == len(counts)
    assert result["dominant_lineage_id"] == (0 if counts else None)
    assert result["dominant_lineage_fraction"] == (max(counts) / sum(counts) if counts else None)
    assert result["dominant_lineage_tie_count"] == (counts.count(max(counts)) if counts else 0)
    assert result["shannon_entropy"] == (pytest.approx(entropy) if counts else None)
    assert result["simpson_concentration"] == (pytest.approx(concentration) if counts else None)
    assert result["lineage_evenness"] == (pytest.approx(evenness) if evenness else None)
    assert result["simpson_diversity"] == (
        None if concentration is None else pytest.approx(1 - concentration)
    )
    assert result["effective_lineages_shannon"] == (
        None if entropy is None else pytest.approx(np.exp(entropy))
    )
    assert result["effective_lineages_simpson"] == (
        None if concentration is None else pytest.approx(1 / concentration)
    )


def test_distance_and_pairwise_variance_identity() -> None:
    a, b = np.array([0.0, 0.0]), np.array([3.0, 4.0])
    distance = vector_distance(a, b)
    assert distance.l1_distance == 7
    assert distance.mean_absolute_parameter_difference == 3.5
    assert distance.l2_distance == 5
    assert distance.rms_parameter_difference == sqrt(12.5)
    assert distance.maximum_absolute_parameter_difference == 4
    vectors = np.stack([a, b])
    variance = float(vectors.var(axis=0).mean())
    assert variance == 3.125
    assert _pairwise_rms(variance, len(vectors)) == sqrt(12.5)
    triples = np.stack([a, b, a + 1.0])
    distances = [
        vector_distance(triples[i], triples[j]).rms_parameter_difference ** 2
        for i, j in ((0, 1), (0, 2), (1, 2))
    ]
    assert _pairwise_rms(float(triples.var(axis=0).mean()), len(triples)) == pytest.approx(
        sqrt(float(np.mean(distances)))
    )


@pytest.mark.parametrize(
    ("vectors", "expected_within", "expected_between"),
    [
        ([[0.0], [2.0], [0.0], [2.0]], 1.0, 0.0),
        ([[0.0], [0.0], [2.0], [2.0]], 0.0, 1.0),
        ([[0.0], [2.0], [2.0], [4.0]], 1.0, 1.0),
    ],
)
def test_variance_decomposition(
    vectors: list[list[float]], expected_within: float, expected_between: float
) -> None:
    total, within, between, _ = _variance_components(np.asarray(vectors), np.array([1, 1, 2, 2]))
    assert within == pytest.approx(expected_within)
    assert between == pytest.approx(expected_between)
    assert total == pytest.approx(within + between)


def test_compensating_mutations_and_missing_genomes() -> None:
    run = _forest()
    assert ancestral_mutation_totals(run, 4) == {
        "path_effective_mutations": 2,
        "path_sum_abs_delta": 2.0,
    }
    result = retained_genome_diversity(run, [10, 14, 15])
    assert result["genomes_available"] == 3
    assert result["distinct_genome_contents"] == 1
    assert result["mean_pairwise_l2_distance"] == 0
    none_available = retained_genome_diversity(run, [999])
    assert none_available["quality"] is Quality.UNAVAILABLE
    assert none_available["unavailable_genome_ids"] == [999]
    assert none_available["mean_pairwise_l2_distance"] is None
    with pytest.raises(ValueError, match="exceeds max_genomes"):
        retained_genome_diversity(run, [10, 14], max_genomes=1)
    with pytest.raises(ValueError, match="no observed birth edge"):
        parent_child_genome_divergence(run, 0)
    with pytest.raises(ValueError, match="equally sized"):
        vector_distance(np.zeros(1), np.zeros(2))
    two_lineages = pl.DataFrame({"step": [0, 0], "lineage_id": [0, 1], "population_after": [1, 1]})
    assert _diversity(two_lineages, 0, 1)["active_lineages"].item() == 2
    zero = run.run.load_highlighted_genome(10)
    final = run.run.load_highlighted_genome(14)
    assert vector_distance(flatten_genome(zero), flatten_genome(final)).l2_distance == 0
    assert parent_child_genome_divergence(run, 2)["quality"] is Quality.UNAVAILABLE
    assert get_ancestors(run, 2)["agent_id"].to_list() == [0]
    vector = flatten_genome(run.run.load_highlighted_genome(10))
    assert vector.shape == (POLICY_PARAMETER_COUNT,)
    with pytest.raises(ValueError, match="incompatible genome leaf"):
        flatten_genome(
            tuple(np.zeros(spec.shape, dtype=np.float64) for spec in POLICY_PARAMETER_SPECS)
        )  # pyright: ignore[reportArgumentType]


def test_lineage_queries_and_recorded_mutation_opportunities() -> None:
    run = _forest()
    fractions = lineage_proportions_at_step(run, 3).collect()
    assert fractions["population_after"].to_list() == [3]
    assert fractions["population_fraction"].to_list() == [1.0]
    with pytest.raises(ValueError, match="exceeds max_rows"):
        lineage_population_series(run, dense=True, max_rows=1)
    assert lineage_population_series(run, lineage_ids=[99]).collect().is_empty()

    curve = survival_curve_by_lineage(run, lineage_ids=[0]).collect()
    assert curve["death_count"].sum() == 1
    assert curve["censored_count"].sum() == 4
    assert curve["survival_probability"].min() == pytest.approx(0.8)

    mutation_row = mutation_summary_by_lineage(run).collect().row(0, named=True)
    assert mutation_row["mutation_selected_total"] == 2
    assert mutation_row["selected_weights"] == 2
    assert mutation_row["selected_biases"] == 0
    assert mutation_row["births_with_effective_mutations"] == 2
    assert mutation_row["total_parameter_opportunities"] == 4 * POLICY_PARAMETER_COUNT
    assert mutation_row["effective_mutation_fraction"] == pytest.approx(
        0.5 / POLICY_PARAMETER_COUNT
    )
    by_generation = mutation_summary_by_generation(run).collect()
    assert by_generation["mutation_effective_total"].to_list() == [0, 1, 1, 0]
    assert by_generation["effective_mutation_fraction"][0] is None
    by_window = mutation_summary_by_window(run, 1).collect()
    assert by_window["births_total"].to_list() == [1, 1, 1, 0, 1, 0]
    assert by_window["mean_abs_delta_per_birth"].to_list() == [1, 0, 1, None, 0, None]
