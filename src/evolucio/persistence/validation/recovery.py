"""Check the imported prefix and, when available, its source chain."""

from evolucio.persistence.loaders.errors import DataConsistencyError
from evolucio.persistence.loaders.run_data import RunData, read_run_metadata
from evolucio.persistence.local.artifacts import artifact_path
from evolucio.persistence.parquet.index import prefix_digest
from evolucio.persistence.zarr.storage import verify_store_tree
from evolucio.recovery.errors import HistoryIntegrityError
from evolucio.recovery.history import prefix_frames


def check_recovery(run: RunData) -> bool:
    provenance = run.recovery
    assert provenance is not None
    index = run.parquet
    if index is None or index.schema_version != 2:
        raise DataConsistencyError("Recovery requires committed telemetry")
    step = provenance.inherited_steps
    counts = dict.fromkeys(index.datasets, 0)
    for chunk in index.chunks:
        if chunk.end_step_exclusive > step:
            break
        for name in counts:
            counts[name] += chunk.counts[name]
    if tuple(counts[name] for name in ("metrics", "births", "deaths", "genealogy")) != (
        provenance.inherited_metrics_rows,
        provenance.inherited_birth_rows,
        provenance.inherited_death_rows,
        provenance.inherited_genealogy_rows,
    ) or (step and not any(chunk.end_step_exclusive == step for chunk in index.chunks)):
        raise DataConsistencyError("Imported row counts or prefix boundary differ")
    seen = {run.manifest.run_id}
    child = run
    evidence_complete = True
    while (link := child.recovery) is not None:
        if link.source_run_id in seen:
            raise DataConsistencyError("Cyclic recovery provenance")
        seen.add(link.source_run_id)
        source_path = run.path.parent / link.source_run_id
        if source_path.is_symlink():
            raise DataConsistencyError("Symlink in recovery source path")
        if not source_path.exists():
            return False
        source = read_run_metadata(source_path)
        source_checkpoints = source.list_checkpoints()
        parent = source.recovery
        if (
            source.manifest.config_hash != link.source_config_hash
            or link.recovery_depth != (parent.recovery_depth + 1 if parent else 1)
            or link.root_run_id != (parent.root_run_id if parent else source.manifest.run_id)
            or source.parquet is None
            or child.parquet is None
            or prefix_digest(source.parquet, link.inherited_steps) != link.source_prefix_digest
            or any(
                capture.state_step == link.source_checkpoint_step
                and capture.sha256 != link.source_checkpoint_hash
                for capture in source_checkpoints
            )
        ):
            raise DataConsistencyError(
                "Recovery source metadata disagrees with the imported prefix"
            )
        matching = [c for c in source_checkpoints if c.state_step == link.source_checkpoint_step]
        if matching:
            capture = matching[0]
            if verify_store_tree(artifact_path(source.path, capture.relative_path)) != (
                capture.byte_size,
                link.source_checkpoint_hash,
            ):
                raise DataConsistencyError("Source checkpoint bytes disagree with provenance")
        else:
            evidence_complete = False
        try:
            children = prefix_frames(child.path, child.parquet, link.inherited_steps)
            parents = prefix_frames(source.path, source.parquet, link.inherited_steps)
            for (imported_chunk, imported), (source_chunk, original) in zip(
                children, parents, strict=True
            ):
                if imported_chunk != source_chunk or any(
                    not imported[name].drop("run_id").equals(frame.drop("run_id"))
                    for name, frame in original.items()
                ):
                    raise DataConsistencyError("Imported chunk differs from the source")
        except HistoryIntegrityError as exc:
            raise DataConsistencyError("Invalid committed history in the provenance chain") from exc
        if not child.scan_founders().collect().equals(source.scan_founders().collect()):
            raise DataConsistencyError("Imported founders differ from the source")
        child = source
    return evidence_complete
