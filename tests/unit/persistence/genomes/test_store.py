"""Selected policy hash, deduplication and integrity checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from evolucio.cli.main import main
from evolucio.core.policy.schema import POLICY_PARAMETER_SPECS
from evolucio.persistence.genomes import (
    INDEX_PATH,
    GenomeHighlightReason,
    GenomeIndex,
    GenomeLeaves,
    HighlightedGenomeRecord,
    genome_content_hash,
    load_highlighted_genome,
    store_genome,
    verify_genome_store,
)
from evolucio.persistence.local.artifacts import sha256_file


def test_identical_bytes_deduplicate_and_verify(tmp_path: Path) -> None:
    leaves: GenomeLeaves = tuple(
        np.zeros(spec.shape, dtype=np.float32) for spec in POLICY_PARAMETER_SPECS
    )
    digest, relative, sha, size = store_genome(tmp_path, leaves)
    assert store_genome(tmp_path, tuple(leaf.copy() for leaf in leaves)) == (
        digest,
        relative,
        sha,
        size,
    )
    records = tuple(
        HighlightedGenomeRecord(
            genome_id=key,
            genome_hash=digest,
            agent_id=key,
            parent_genome_id=None,
            lineage_id=key,
            generation=0,
            reasons={GenomeHighlightReason.FOUNDER: 0},
            relative_path=relative,
            artifact_sha256=sha,
            byte_size=size,
        )
        for key in (1, 2)
    )
    index = GenomeIndex(run_id="run", genomes=records)
    index_path = tmp_path / INDEX_PATH
    index_path.write_text(index.model_dump_json())
    files = list((tmp_path / "artifacts/genomes").glob("*.npz"))
    assert files == [tmp_path / relative]
    assert verify_genome_store(tmp_path).ok

    orphan = files[0].with_name("orphan.npz")
    orphan.write_bytes(files[0].read_bytes())
    assert any("orphan" in error for error in verify_genome_store(tmp_path).errors)
    orphan.unlink()

    path = files[0]
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    assert any("SHA" in error for error in verify_genome_store(tmp_path).errors)
    path.write_bytes(original)
    w1, b1, w2, b2 = leaves
    np.savez(path, layer1_weight=w1 + 1, layer1_bias=b1, layer2_weight=w2, layer2_bias=b2)
    bad_records = tuple(
        record.model_copy(
            update={
                "artifact_sha256": sha256_file(path),
                "byte_size": path.stat().st_size,
            }
        )
        for record in records
    )
    index_path.write_text(index.model_copy(update={"genomes": bad_records}).model_dump_json())
    assert any("semantic genome hash" in error for error in verify_genome_store(tmp_path).errors)
    with pytest.raises(ValueError, match="semantic genome hash"):
        store_genome(tmp_path, leaves)


def test_invalid_dtype_and_layout() -> None:
    leaves: GenomeLeaves = tuple(
        np.zeros(spec.shape, dtype=np.float32) for spec in POLICY_PARAMETER_SPECS
    )
    with pytest.raises(ValueError, match="dtype"):
        genome_content_hash((leaves[0].astype(np.float64), *leaves[1:]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="C-order"):
        genome_content_hash((np.asfortranarray(leaves[0]), *leaves[1:]))


def test_index_checks_metadata_run_and_references(tmp_path: Path) -> None:
    leaves: GenomeLeaves = tuple(
        np.zeros(spec.shape, dtype=np.float32) for spec in POLICY_PARAMETER_SPECS
    )
    assert not verify_genome_store(tmp_path).ok
    assert main(["genome", "list", str(tmp_path)]) == 2
    assert main(["genome", "verify", str(tmp_path)]) == 1
    digest, relative, sha, size = store_genome(tmp_path, leaves)
    record = HighlightedGenomeRecord(
        genome_id=1,
        genome_hash=digest,
        agent_id=1,
        parent_genome_id=None,
        lineage_id=1,
        generation=0,
        reasons={GenomeHighlightReason.FOUNDER: 0},
        relative_path=relative,
        artifact_sha256=sha,
        byte_size=size,
    )
    index = GenomeIndex(run_id="test-run", genomes=(record,))
    index_path = tmp_path / INDEX_PATH
    index_path.write_text(index.model_dump_json())
    with pytest.raises(KeyError):
        load_highlighted_genome(tmp_path, 2)

    for options, error in (
        ({"relative_path": "different.npz"}, "content path"),
        ({"policy_schema_version": 2}, "incompatible policy"),
        ({"reasons": {"founder": -1}}, "nonnegative"),
        ({"generation": 1}, "ancestry"),
        ({"generation": 1, "parent_genome_id": 0}, "generation zero"),
    ):
        payload = record.model_dump(mode="json") | options
        index_path.write_text(json.dumps(index.model_dump(mode="json") | {"genomes": [payload]}))
        assert any(error in message for message in verify_genome_store(tmp_path).errors)

    for options, error in (
        ({"genome_id": 1, "agent_id": 2}, "duplicate genome"),
        ({"genome_id": 2, "agent_id": 1}, "duplicate highlighted agent"),
        ({"genome_id": 2, "agent_id": 2, "byte_size": size + 1}, "incompatible references"),
    ):
        copied = record.model_dump(mode="json") | options
        index_path.write_text(
            json.dumps(
                index.model_dump(mode="json")
                | {
                    "genomes": [record.model_dump(mode="json"), copied],
                }
            )
        )
        assert any(error in message for message in verify_genome_store(tmp_path).errors)

    representative = record.model_dump(mode="json") | {
        "reasons": {"final_lineage_representative": 3}
    }
    index_path.write_text(
        json.dumps(
            index.model_dump(mode="json")
            | {
                "genomes": [representative, representative | {"genome_id": 2, "agent_id": 2}],
            }
        )
    )
    assert any(
        "multiple final representatives" in error for error in verify_genome_store(tmp_path).errors
    )
    summary = index.model_dump(mode="json")
    del summary["schema_version"]
    index_path.write_text(json.dumps(summary))
    assert any("schema metadata" in error for error in verify_genome_store(tmp_path).errors)

    index_path.write_text(index.model_dump_json())
    (tmp_path / relative).unlink()
    assert any(relative in error for error in verify_genome_store(tmp_path).errors)
