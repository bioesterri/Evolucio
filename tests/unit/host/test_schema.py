import hashlib
import json
from dataclasses import fields

from evolucio.host import (
    HOST_DTO_SCHEMA_DIGEST,
    HOST_DTO_SCHEMA_NAME,
    HOST_DTO_SCHEMA_VERSION,
    StepMetricsDTO,
    host_dto_schema_digest,
    host_dto_schema_payload,
)


def test_schema_canonical_digest_and_contract() -> None:
    payload = host_dto_schema_payload()
    assert payload["name"] == HOST_DTO_SCHEMA_NAME == "host_metrics_events_snapshots_v1"
    assert payload["version"] == HOST_DTO_SCHEMA_VERSION == 1
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == HOST_DTO_SCHEMA_DIGEST
    assert host_dto_schema_digest() == HOST_DTO_SCHEMA_DIGEST
    assert payload["fields"]["StepMetricsDTO"] == [field.name for field in fields(StepMetricsDTO)]
    assert payload["snapshot"]["resumable"] is False
    assert payload["snapshot"]["rng"] is False
    assert payload["snapshot"]["full_genomes"] is False
    assert payload["in_compile_signature"] is False
    assert payload["in_scientific_config_hash"] is False
    payload["snapshot"]["rng"] = True
    assert host_dto_schema_payload()["snapshot"]["rng"] is False
