"""Scientific checks over committed events, projections and captured states."""

from collections.abc import Callable
from math import isclose

import numpy as np
import polars as pl

from evolucio.analysis.summaries import (
    AGENTS_PATH,
    LINEAGES_PATH,
    SURVIVOR_SCHEMA,
    build_agent_summaries,
    build_lineage_summaries,
)
from evolucio.core.codes import DeathCauseCode
from evolucio.core.metrics.events import DeathPhaseCode
from evolucio.core.metrics.schema import METRICS_SCHEMA_DIGEST
from evolucio.core.simulation.schema import SIMULATION_STEP_SCHEMA_DIGEST
from evolucio.persistence.genomes import GenomeHighlightReason
from evolucio.persistence.loaders.errors import DataSchemaError
from evolucio.persistence.loaders.run_data import RunData
from evolucio.persistence.parquet.schemas import DEATH_CAUSE_COLUMNS, GENEALOGY_COLUMNS
from evolucio.persistence.zarr.checkpoint import read_checkpoint_arrays

type Finding = Callable[[str, str, str, int | None], None]


def row_count(frame: pl.LazyFrame) -> int:
    return int(frame.select(pl.len()).collect(engine="streaming").item())


def check_telemetry(run: RunData, issue: Finding) -> None:
    index = run.parquet
    assert index is not None
    schemas = run.manifest.schemas
    if (schemas.metrics is not None and schemas.metrics != METRICS_SCHEMA_DIGEST) or (
        schemas.simulation_step is not None
        and schemas.simulation_step != SIMULATION_STEP_SCHEMA_DIGEST
    ):
        raise DataSchemaError("Scientific checks require recognized metric and model definitions")
    metrics, births, deaths, edges = (
        run.scan_metrics(),
        run.scan_births(),
        run.scan_deaths(),
        run.scan_genealogy(),
    )

    def invalid(code: str, frame: pl.LazyFrame, message: str) -> None:
        if count := row_count(frame):
            issue(code, "parquet", message, count)

    for frame, keys in (
        (metrics, ["step"]),
        (births, ["birth_step", "child_agent_id"]),
        (deaths, ["death_step", "agent_id"]),
        (edges, ["birth_step", "child_agent_id"]),
    ):
        if not frame.select(keys).collect().equals(frame.select(keys).sort(keys).collect()):
            issue("noncanonical_order", "parquet", f"Rows are out of order for {keys}", None)
    for frame, key in (
        (metrics, "step"),
        (births, "child_agent_id"),
        (births, "child_genome_id"),
        (deaths, "agent_id"),
        (edges, "child_agent_id"),
    ):
        invalid("duplicate_identity", frame.group_by(key).len().filter(pl.col("len") > 1), key)
    for left, right in (
        (births.select(GENEALOGY_COLUMNS), edges),
        (edges, births.select(GENEALOGY_COLUMNS)),
    ):
        invalid(
            "genealogy_projection",
            left.join(right, on=list(GENEALOGY_COLUMNS), how="anti"),
            "Births and genealogy encode different parent edges",
        )
    interval = index.policy.metrics_interval
    invalid(
        "metrics_sampling",
        metrics.filter(pl.col("step") % interval != 0),
        "Metric step falls outside the sampling schedule",
    )
    for events, step, count in (
        (births, "birth_step", "birth_count"),
        (deaths, "death_step", "death_count"),
    ):
        observed = events.group_by(step).agg(pl.len().alias("observed"))
        invalid(
            "event_count",
            metrics.join(observed, left_on="step", right_on=step, how="left").filter(
                pl.col(count) != pl.col("observed").fill_null(0)
            ),
            f"Sampled {count} disagrees with events",
        )
    cause_counts = deaths.group_by("death_step").agg(
        *(
            (pl.col("cause_code") == code).sum().alias(f"observed_{column}")
            for code, column in enumerate(DEATH_CAUSE_COLUMNS)
        )
    )
    invalid(
        "metric_death_causes",
        metrics.join(cause_counts, left_on="step", right_on="death_step", how="left").filter(
            pl.any_horizontal(
                *(
                    pl.col(column) != pl.col(f"observed_{column}").fill_null(0)
                    for column in DEATH_CAUSE_COLUMNS
                )
            )
        ),
        "Cause counts disagree with death events",
    )
    cause_codes = [int(code) for code in DeathCauseCode if code != DeathCauseCode.NONE]
    phase_codes = [int(DeathPhaseCode.PRE_ACTION), int(DeathPhaseCode.POST_ACTION)]
    invalid(
        "death_payload",
        deaths.filter(
            ~pl.col("cause_code").is_in(cause_codes)
            | ~pl.col("phase_code").is_in(phase_codes)
            | (pl.col("age") < 0)
            | (
                (pl.col("cause_code") != int(DeathCauseCode.INVALID_STATE))
                & (
                    ~pl.col("energy").is_finite()
                    | (pl.col("position_x") < 0)
                    | (pl.col("position_y") < 0)
                    | (pl.col("position_x") >= run.config.world.width)
                    | (pl.col("position_y") >= run.config.world.height)
                )
            )
        ),
        "Invalid death cause, phase or payload",
    )
    invalid(
        "metric_values",
        metrics.filter(
            ~pl.col("alive_count").is_between(0, run.config.population.max_agents)
            | ~pl.col("active_lineage_count").is_between(0, pl.col("alive_count"))
            | pl.any_horizontal(
                pl.col(
                    "birth_count",
                    "death_count",
                    "total_resources",
                    "total_energy_alive",
                    "mean_energy_alive",
                    "genetic_parameter_variance",
                    *DEATH_CAUSE_COLUMNS,
                )
                < 0
            )
            | pl.any_horizontal(
                ~pl.col(
                    "total_resources",
                    "total_energy_alive",
                    "mean_energy_alive",
                    "genetic_parameter_variance",
                ).is_finite()
            )
            | (pl.sum_horizontal(*DEATH_CAUSE_COLUMNS) != pl.col("death_count"))
        ),
        "Invalid population, aggregate or mortality metric",
    )


def check_population(run: RunData, issue: Finding) -> None:
    founders, births, deaths = run.scan_founders(), run.scan_births(), run.scan_deaths()
    initial = row_count(founders)
    if initial != run.config.population.initial_agents:
        issue("founder_count", "population", "Founder count differs from frozen config", None)
    if row_count(
        founders.filter(
            (pl.col("generation") != 0)
            | (pl.col("birth_step") != 0)
            | (pl.col("lineage_id") != pl.col("agent_id"))
        )
    ):
        issue("founder_identity", "population", "Invalid founder generation, step or lineage", None)
    columns = ["agent_id", "genome_id", "lineage_id", "generation", "birth_step", "parent_id"]
    origins = pl.concat(
        [
            founders.with_columns(pl.lit(-1, dtype=pl.Int64).alias("parent_id")).select(columns),
            births.rename(
                {
                    "child_agent_id": "agent_id",
                    "child_genome_id": "genome_id",
                    "parent_agent_id": "parent_id",
                }
            ).select(columns),
        ]
    )
    for identity in ("agent_id", "genome_id"):
        duplicates = row_count(origins.group_by(identity).len().filter(pl.col("len") > 1))
        if duplicates:
            issue("duplicate_origin", "population", identity, duplicates)
    parents = origins.select(
        pl.col("agent_id").alias("parent_agent_id"),
        *(pl.col(name).alias(f"expected_{name}") for name in columns if name != "agent_id"),
    )
    invalid_parents = (
        births.join(parents, on="parent_agent_id", how="left")
        .join(
            deaths.select(pl.col("agent_id").alias("parent_agent_id"), "death_step"),
            on="parent_agent_id",
            how="left",
        )
        .filter(
            pl.col("expected_genome_id").is_null()
            | (pl.col("parent_genome_id") != pl.col("expected_genome_id"))
            | (pl.col("lineage_id") != pl.col("expected_lineage_id"))
            | (pl.col("generation") != pl.col("expected_generation") + 1)
            | (
                (pl.col("expected_generation") > 0)
                & (pl.col("birth_step") <= pl.col("expected_birth_step"))
            )
            | (pl.col("death_step") <= pl.col("birth_step"))
        )
    )
    if count := row_count(invalid_parents):
        issue(
            "invalid_parent",
            "population",
            "Parent identity, lineage, age or lifecycle disagrees",
            count,
        )
    invalid_deaths = deaths.join(origins, on="agent_id", how="left", suffix="_origin").filter(
        pl.col("birth_step").is_null()
        | ((pl.col("generation_origin") > 0) & (pl.col("death_step") <= pl.col("birth_step")))
        | pl.any_horizontal(
            *(
                pl.col(name) != pl.col(f"{name}_origin")
                for name in ("parent_id", "lineage_id", "genome_id", "generation")
            )
        )
    )
    if count := row_count(invalid_deaths):
        issue("invalid_death_origin", "population", "Death has no matching living origin", count)
    delta = (
        pl.concat(
            [
                births.select(
                    pl.col("birth_step").alias("step"), pl.lit(1, dtype=pl.Int64).alias("delta")
                ),
                deaths.select(
                    pl.col("death_step").alias("step"), pl.lit(-1, dtype=pl.Int64).alias("delta")
                ),
                run.scan_metrics().select("step", pl.lit(0, dtype=pl.Int64).alias("delta")),
            ]
        )
        .group_by("step")
        .agg(pl.col("delta").sum())
        .sort("step")
        .with_columns((pl.col("delta").cum_sum() + initial).alias("expected_alive"))
    )
    if count := row_count(
        delta.filter(~pl.col("expected_alive").is_between(0, run.config.population.max_agents))
    ):
        issue(
            "population_capacity", "population", "Event balance exceeds population capacity", count
        )
    if count := row_count(
        run.scan_metrics()
        .join(delta, on="step")
        .filter(pl.col("alive_count") != pl.col("expected_alive"))
    ):
        issue(
            "population_balance",
            "population",
            "Sampled population differs from event balance",
            count,
        )
    genome_keys = ["agent_id", "genome_id", "lineage_id", "generation"]
    highlights = pl.DataFrame(
        [
            (r.agent_id, r.genome_id, r.lineage_id, r.generation)
            for r in run.list_highlighted_genomes()
        ],
        schema=dict.fromkeys(genome_keys, pl.Int64),
        orient="row",
    )
    if count := row_count(highlights.lazy().join(origins, on=genome_keys, how="anti")):
        issue("genome_origin", "population", "Highlighted genome has no matching origin", count)
    if run.genomes is not None:
        expected = set(founders.select("agent_id", "genome_id", "lineage_id").collect().iter_rows())
        observed = {
            (r.agent_id, r.genome_id, r.lineage_id)
            for r in run.genomes.genomes
            if GenomeHighlightReason.FOUNDER in r.reasons
        }
        if expected != observed:
            issue(
                "founder_genomes",
                "population",
                "Founder genomes differ from founder identities",
                len(expected ^ observed),
            )


def check_summaries(run: RunData, issue: Finding) -> None:
    assert run.summary is not None and run.manifest.final_step is not None
    step = run.manifest.final_step
    paths = {r.relative_path for r in run.manifest.artifacts}
    births, deaths = run.scan_births(), run.scan_deaths()
    if paths.issuperset({AGENTS_PATH, LINEAGES_PATH}):
        agents = run.scan_agent_summaries().collect()
        survivors = agents.lazy().filter(pl.col("alive_at_end")).select(SURVIVOR_SCHEMA.names())
        expected = build_agent_summaries(
            run.manifest.run_id, run.scan_founders(), births, deaths, survivors, step
        )
        if not agents.equals(expected):
            issue(
                "agent_summary", "summaries", "Agent summaries differ from lifecycle records", None
            )
        lineages = run.scan_lineage_summaries().collect()
        expected_lineages = build_lineage_summaries(expected, births, step)
        if not lineages.equals(expected_lineages):
            issue(
                "lineage_summary",
                "summaries",
                "Lineage summaries differ from agents and births",
                None,
            )
        final_alive = agents.filter(pl.col("alive_at_end"))
        invalid_ages = final_alive.filter(
            pl.col("final_age")
            != (step - pl.col("birth_step") - (pl.col("generation") > 0).cast(pl.Int64))
        ).height
        if invalid_ages:
            issue(
                "agent_final_age",
                "summaries",
                "Surviving agent age disagrees with origin",
                invalid_ages,
            )
        compare_summary(
            run,
            issue,
            {
                "alive_count": final_alive.height,
                "active_lineage_count": final_alive["lineage_id"].n_unique(),
                "final_total_energy": float(final_alive["final_energy"].sum()),
            },
        )
    compare_summary(
        run,
        issue,
        {"alive_count": row_count(run.scan_founders()) + row_count(births) - row_count(deaths)},
    )
    saturated = False
    for record in run.list_checkpoints():
        if record.state_step == step:
            _, arrays = read_checkpoint_arrays(run.path / record.relative_path)
            saturated = bool(arrays["accumulator/count_overflow"].item())
    if not saturated:
        compare_summary(
            run, issue, {"births_total": row_count(births), "deaths_total": row_count(deaths)}
        )
    terminal = run.scan_metrics().filter(pl.col("step") == step - 1).collect()
    if terminal.height == 1:
        compare_summary(
            run,
            issue,
            {
                "alive_count": terminal["alive_count"][0],
                "active_lineage_count": terminal["active_lineage_count"][0],
                "final_total_energy": terminal["total_energy_alive"][0],
                "final_total_resources": terminal["total_resources"][0],
            },
        )


def compare_summary(run: RunData, issue: Finding, expected: dict[str, int | float]) -> None:
    assert run.summary is not None
    for key, value in expected.items():
        actual = run.summary.get(key)
        matches = (
            type(actual) is int and actual == value
            if isinstance(value, int)
            else isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and isclose(actual, value, rel_tol=1e-6, abs_tol=1e-6)
        )
        if not matches:
            issue("final_summary", "summaries", f"{key} disagrees with persisted evidence", 1)


def compare_capture(
    run: RunData,
    issue: Finding,
    component: str,
    step: int,
    alive_count: int,
    lineages: int,
    energy: float,
    resources: float,
) -> bool:
    if run.parquet is None:
        return False
    row = run.scan_metrics().filter(pl.col("step") == step - 1).collect()
    if row.height != 1:
        return False
    if (
        alive_count != row["alive_count"][0]
        or lineages != row["active_lineage_count"][0]
        or not np.isclose(energy, row["total_energy_alive"][0], rtol=1e-6, atol=1e-6)
        or not np.isclose(resources, row["total_resources"][0], rtol=1e-6, atol=1e-6)
    ):
        issue("capture_metrics", component, f"State {step} disagrees with metric {step - 1}", 1)
    return True
