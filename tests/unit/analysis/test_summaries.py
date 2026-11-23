"""Lifecycle and lineage invariants using compact host frames."""

import polars as pl
import pytest

from evolucio.analysis.summaries import (
    FOUNDER_SCHEMA,
    SURVIVOR_SCHEMA,
    build_agent_summaries,
    build_lineage_summaries,
    select_final_representatives,
)
from evolucio.persistence.parquet.schemas import BIRTHS_SCHEMA, DEATHS_SCHEMA


def test_leaders_generation_and_extinction() -> None:
    founders = pl.DataFrame(
        [(1, 1, 1, 0, 0, 10.0, 0, 0), (7, 7, 7, 0, 0, 1.0, 0, 0)],
        schema=FOUNDER_SCHEMA,
        orient="row",
    ).lazy()
    births_data = [
        {
            "run_id": "run",
            "birth_step": step,
            "child_agent_id": child,
            "parent_agent_id": parent,
            "lineage_id": 1,
            "generation": generation,
            "child_genome_id": child,
            "parent_genome_id": parent,
            "mutation_selected_count": 2,
            "mutation_effective_count": 1,
            "mutation_sum_abs_delta": 0.125,
        }
        for child, parent, generation, step in [
            (2, 1, 1, 1),
            (3, 1, 1, 2),
            (4, 1, 1, 3),
            (5, 2, 2, 4),
            (6, 2, 2, 5),
        ]
    ]
    births = pl.from_dicts(births_data, schema=BIRTHS_SCHEMA).lazy()
    deaths = pl.from_dicts(
        [
            {
                "run_id": "run",
                "death_step": 39,
                "agent_id": 1,
                "parent_id": -1,
                "lineage_id": 1,
                "genome_id": 1,
                "generation": 0,
                "age": 10,
                "energy": 0.0,
                "cause_code": 1,
                "phase_code": 1,
            },
            {
                "run_id": "run",
                "death_step": 2,
                "agent_id": 7,
                "parent_id": -1,
                "lineage_id": 7,
                "genome_id": 7,
                "generation": 0,
                "age": 2,
                "energy": 0.0,
                "cause_code": 1,
                "phase_code": 1,
            },
        ],
        schema=DEATHS_SCHEMA,
    ).lazy()
    survivors = pl.DataFrame(
        [
            (2, 1, 1, 2, 1, 1, 30, 2.0),
            (3, 1, 1, 3, 1, 2, 30, 2.0),
            (4, 1, 1, 4, 1, 3, 10, 2.0),
            (5, 2, 1, 5, 2, 4, 10, 2.0),
            (6, 2, 1, 6, 2, 5, 10, 2.0),
        ],
        schema=SURVIVOR_SCHEMA,
        orient="row",
    )

    agents = build_agent_summaries("run", founders, births, deaths, survivors.lazy(), 40)
    assert agents.height == 7
    assert agents["direct_offspring_count"].to_list() == [3, 2, 0, 0, 0, 0, 0]
    active, extinct = build_lineage_summaries(agents, births, 40).to_dicts()
    assert active["reproductive_leader_agent_id"] == 1
    assert active["survival_leader_agent_id"] == 2
    assert active["max_generation"] == 2 and active["total_descendants"] == 5
    assert active["mutation_selected_total"] == 10
    assert active["mutation_effective_total"] == 5
    assert active["mutation_abs_delta_total"] == 0.625
    assert active["extinction_step"] is None and active["duration_steps"] == 40
    assert extinct["extinction_step"] == 2 and extinct["duration_steps"] == 3
    assert extinct["final_population_fraction"] == 0.0
    assert select_final_representatives(survivors)["agent_id"].to_list() == [5]

    tied_births = births.filter(pl.col("child_agent_id") != 4)
    tied_survivors = survivors.filter(pl.col("agent_id") != 4)
    tied_agents = build_agent_summaries(
        "run", founders, tied_births, deaths, tied_survivors.lazy(), 40
    )
    assert (
        build_lineage_summaries(tied_agents, tied_births, 40)["reproductive_leader_agent_id"][0]
        == 1
    )

    with pytest.raises(ValueError, match="lifecycle"):
        build_agent_summaries("run", founders, births, deaths, survivors.head(4).lazy(), 40)


def test_total_extinction() -> None:
    founders = pl.DataFrame(
        [(1, 1, 1, 0, 0, 1.0, 0, 0)],
        schema=FOUNDER_SCHEMA,
        orient="row",
    ).lazy()
    empty_births = pl.DataFrame(schema=BIRTHS_SCHEMA).lazy()
    death = pl.DataFrame(
        [("run", 3, 1, -1, 1, 1, 0, 4, 0.0, 0, 0, 1, 1)],
        schema=DEATHS_SCHEMA,
        orient="row",
    ).lazy()
    empty_survivors = pl.DataFrame(schema=SURVIVOR_SCHEMA)
    agents = build_agent_summaries("run", founders, empty_births, death, empty_survivors.lazy(), 4)
    lineage = build_lineage_summaries(agents, empty_births, 4).row(0, named=True)
    assert lineage["alive_final"] == 0 and lineage["final_population_fraction"] == 0
    assert lineage["extinction_step"] == 3 and lineage["duration_steps"] == 4
    assert select_final_representatives(empty_survivors).is_empty()
