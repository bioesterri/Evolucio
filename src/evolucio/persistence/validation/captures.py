"""Cross-check one capture at a time against the committed history."""

import numpy as np
import polars as pl

from evolucio.analysis.summaries import FOUNDERS_PATH
from evolucio.persistence.loaders.run_data import RunData
from evolucio.persistence.parquet.index import prefix_digest
from evolucio.persistence.zarr.checkpoint import read_checkpoint_arrays
from evolucio.persistence.zarr.index import CheckpointRecord, SnapshotRecord

from .scientific import Finding, compare_capture, compare_summary, row_count


def check_captures(run: RunData, component: str, issue: Finding) -> tuple[int, int]:
    records: tuple[CheckpointRecord | SnapshotRecord, ...] = (
        run.list_checkpoints() if component == "checkpoints" else run.list_snapshots()
    )
    compared = 0
    has_founders = (run.manifest.initial_step == 0 or run.recovery is not None) and any(
        r.relative_path == FOUNDERS_PATH for r in run.manifest.artifacts
    )
    for record in records:
        step = record.state_step
        if isinstance(record, CheckpointRecord):
            metadata, arrays = read_checkpoint_arrays(run.path / record.relative_path)
            mask = arrays["population/alive"]
            ids = arrays["population/agent_id"][mask]
            lineages = arrays["population/lineage_id"][mask]
            energy = float(arrays["population/energy"][mask].sum())
            resource_map = arrays["world/resources"]
            resources = float(resource_map.sum())
            positions = arrays["population/position"][mask]
            occupancy = arrays["world/occupancy"]
            if (
                np.any(positions < 0)
                or np.any(positions[:, 0] >= metadata.width)
                or np.any(positions[:, 1] >= metadata.height)
                or np.any(~np.isfinite(arrays["population/energy"][mask]))
                or np.any(~np.isfinite(arrays["world/environment"]))
            ):
                issue("checkpoint_state", component, f"Invalid host state at step {step}", 1)
            else:
                expected_occupancy = np.zeros_like(occupancy)
                np.add.at(expected_occupancy, (positions[:, 1], positions[:, 0]), 1)
                if not np.array_equal(expected_occupancy, occupancy):
                    issue("checkpoint_occupancy", component, f"Invalid occupancy at step {step}", 1)
            capacity, width, height = metadata.capacity, metadata.width, metadata.height
            overflow = bool(arrays["accumulator/count_overflow"].item())
            if overflow:
                issue(
                    "accumulator_saturated",
                    component,
                    f"Accumulator at state step {step} is saturated",
                    None,
                )
            if (
                run.parquet is not None
                and record.telemetry_prefix_digest is not None
                and (prefix_digest(run.parquet, step) != record.telemetry_prefix_digest)
            ):
                issue(
                    "checkpoint_prefix", component, f"State {step} has a different prefix digest", 1
                )
            if run.parquet is not None and has_founders and not overflow:
                for table, column, key in (
                    (run.scan_births(), "birth_step", "births_total"),
                    (run.scan_deaths(), "death_step", "deaths_total"),
                ):
                    if (
                        row_count(table.filter(pl.col(column) < step))
                        != arrays[f"accumulator/{key}"].item()
                    ):
                        issue("checkpoint_totals", component, f"State {step} disagrees on {key}", 1)
        else:
            snapshot = run.load_snapshot(step)
            population, world = snapshot.population, snapshot.world
            ids, lineages = population.agent_id, population.lineage_id
            energy, resources = float(population.energy.sum()), float(world.resources.sum())
            resource_map = world.resources
            capacity, width, height = population.capacity, world.width, world.height
        resource_capacity = run.config.world.resource_capacity
        if (
            np.any(~np.isfinite(resource_map))
            or np.any(resource_map < 0)
            or np.any(
                (resource_map > resource_capacity)
                & ~np.isclose(resource_map, resource_capacity, rtol=1e-6, atol=1e-6)
            )
        ):
            issue("capture_resources", component, f"Invalid resource map at step {step}", 1)
        if (capacity, width, height) != (
            run.config.population.max_agents,
            run.config.world.width,
            run.config.world.height,
        ) or not 0 <= step <= (run.manifest.initial_step + run.manifest.total_steps):
            issue("capture_context", component, f"State {step} disagrees with the run plan", 1)
        active = len(np.unique(lineages))
        compared += compare_capture(
            run, issue, component, step, len(ids), active, energy, resources
        )
        if step == run.manifest.final_step and run.summary and "alive_count" in run.summary:
            compare_summary(
                run,
                issue,
                {
                    "alive_count": len(ids),
                    "active_lineage_count": active,
                    "final_total_energy": energy,
                    "final_total_resources": resources,
                },
            )
        if has_founders and run.parquet is not None:
            origins = pl.concat(
                [
                    run.scan_founders().select("agent_id"),
                    run.scan_births()
                    .filter(pl.col("birth_step") < step)
                    .select(pl.col("child_agent_id").alias("agent_id")),
                ]
            )
            alive = (
                origins.join(
                    run.scan_deaths().filter(pl.col("death_step") < step).select("agent_id"),
                    on="agent_id",
                    how="anti",
                )
                .select("agent_id")
                .sort("agent_id")
                .collect()
            )
            if alive["agent_id"].to_list() != sorted(ids.tolist()):
                issue(
                    "capture_population",
                    component,
                    f"State {step} has different living agent identities",
                    1,
                )
    return compared, len(records)
