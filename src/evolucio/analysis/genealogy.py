"""Historical identity and bounded traversal of the validated asexual forest."""

from collections import deque
from dataclasses import dataclass

import polars as pl

from evolucio.persistence.loaders.run_data import RunData

from .types import AnalysisUnavailableError, positive_int
from .windows import AnalysisRun, validated


def genealogy_data(run: RunData | AnalysisRun) -> AnalysisRun:
    data = validated(run)
    if not data.population_available:
        raise AnalysisUnavailableError("validated founder census and genealogy required")
    return data


def genealogy_nodes(run: RunData | AnalysisRun) -> pl.LazyFrame:
    data = genealogy_data(run)
    fields = ("lineage_id", "genome_id", "generation", "birth_step")
    founders = data.run.scan_founders().select(
        "agent_id", pl.lit(None, dtype=pl.Int64).alias("parent_agent_id"), *fields
    )
    births = data.run.scan_births().select(
        pl.col("child_agent_id").alias("agent_id"),
        "parent_agent_id",
        "lineage_id",
        pl.col("child_genome_id").alias("genome_id"),
        "generation",
        "birth_step",
    )
    return (
        pl.concat([founders, births], how="vertical_relaxed")
        .join(data.run.scan_deaths().select("agent_id", "death_step"), on="agent_id", how="left")
        .with_columns(pl.lit(data.run.manifest.run_id).alias("run_id"))
        .sort("agent_id")
    )


def genealogy_edges(run: RunData | AnalysisRun) -> pl.LazyFrame:
    return genealogy_data(run).run.scan_genealogy()


def get_ancestors(run: RunData | AnalysisRun, agent_id: int) -> pl.DataFrame:
    """Return the parent through founder, excluding the queried agent."""
    nodes = (
        genealogy_nodes(run)
        .select("agent_id", "parent_agent_id", "genome_id", "generation", "birth_step")
        .collect()
    )
    parents = dict(zip(nodes["agent_id"], nodes["parent_agent_id"], strict=True))
    if agent_id not in parents:
        raise ValueError(f"unknown agent_id {agent_id}")
    chain: list[int] = []
    current = parents[agent_id]
    while current is not None:
        chain.append(current)
        current = parents[current]
    return pl.DataFrame(
        {
            "agent_id": pl.Series(chain, dtype=nodes.schema["agent_id"]),
            "distance_from_agent": pl.Series(range(1, len(chain) + 1), dtype=pl.Int64),
        }
    ).join(nodes, on="agent_id", maintain_order="left")


@dataclass(frozen=True, slots=True)
class DescendantQuery:
    descendants: pl.DataFrame
    partial: bool
    direct_offspring_count: int
    total_descendant_count: int | None


def get_descendants(
    run: RunData | AnalysisRun,
    agent_id: int,
    max_depth: int | None = None,
    max_nodes: int = 10_000,
) -> DescendantQuery:
    positive_int(max_nodes, "max_nodes")
    if max_depth is not None and (type(max_depth) is not int or max_depth < 0):
        raise ValueError("max_depth must be a nonnegative integer")
    nodes = genealogy_nodes(run).select("agent_id", "parent_agent_id", "generation").collect()
    if agent_id not in nodes["agent_id"]:
        raise ValueError(f"unknown agent_id {agent_id}")
    children: dict[int, list[tuple[int, int]]] = {}
    for child, parent, generation in nodes.iter_rows():
        if parent is not None:
            children.setdefault(parent, []).append((child, generation))
    direct = len(children.get(agent_id, ()))
    pending = deque([(agent_id, 0)])
    rows: list[tuple[int, int, int, int]] = []
    partial = False
    while pending:
        parent, depth = pending.popleft()
        for child, generation in children.get(parent, ()):
            if (max_depth is not None and depth >= max_depth) or len(rows) == max_nodes:
                partial = True
                break
            rows.append((child, parent, generation, depth + 1))
            pending.append((child, depth + 1))
        if len(rows) == max_nodes and partial:
            break
    result = pl.DataFrame(
        rows,
        orient="row",
        schema={
            "descendant_agent_id": pl.Int64,
            "parent_agent_id": pl.Int64,
            "generation": pl.Int64,
            "distance_from_ancestor": pl.Int64,
        },
    )
    return DescendantQuery(result, partial, direct, None if partial else len(rows))
