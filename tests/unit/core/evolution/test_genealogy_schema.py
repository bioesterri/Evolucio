import dataclasses
import re

from evolucio.config import build_compile_signature, load_config
from evolucio.core.evolution import (
    GENEALOGY_SCHEMA_DIGEST,
    GENEALOGY_SCHEMA_NAME,
    GENEALOGY_SCHEMA_VERSION,
    genealogy_schema_digest,
    genealogy_schema_payload,
)


def test_genealogy_schema_payload_and_frozen_digest() -> None:
    payload = genealogy_schema_payload()
    assert GENEALOGY_SCHEMA_NAME == "founder_lineage_parent_edge_birth_event_v1"
    assert GENEALOGY_SCHEMA_VERSION == 1
    assert payload["founders"] == {
        "parent_id": "NULL_ID",
        "generation": 0,
        "lineage": "one newly allocated lineage_id per founder",
    }
    assert payload["population_state"] == {"offspring_count": False}
    assert payload["identifiers"]["core_event_id"] is False
    assert payload["birth_event_batch"]["active_row_limit"] == (
        "at most max_births_per_step born rows"
    )
    assert genealogy_schema_digest() == GENEALOGY_SCHEMA_DIGEST
    assert (
        GENEALOGY_SCHEMA_DIGEST
        == "994df298c0d7bbed2f95f206b50b38fe6fe2ee8b12868591c4e7acd479fea271"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", GENEALOGY_SCHEMA_DIGEST)

    signature = build_compile_signature(load_config("tests/fixtures/config/valid_v1.yaml"))
    assert signature.genealogy_schema_version == GENEALOGY_SCHEMA_VERSION
    assert signature.genealogy_schema_digest == GENEALOGY_SCHEMA_DIGEST
    assert not any("event" in field.name for field in dataclasses.fields(signature))
