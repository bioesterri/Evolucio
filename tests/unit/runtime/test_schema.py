import hashlib
import json

from evolucio.runtime import (
    RUNTIME_SCHEMA_DIGEST,
    RUNTIME_SCHEMA_NAME,
    RUNTIME_SCHEMA_VERSION,
    runtime_schema_digest,
    runtime_schema_payload,
)


def test_versioned_canonical_digest() -> None:
    payload = runtime_schema_payload()
    assert payload["name"] == RUNTIME_SCHEMA_NAME == "host_chunked_run_runtime_v1"
    assert payload["version"] == RUNTIME_SCHEMA_VERSION == 1
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    assert runtime_schema_digest() == RUNTIME_SCHEMA_DIGEST
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == RUNTIME_SCHEMA_DIGEST
    for field in (
        "retained_history",
        "early_stop",
        "persistence",
        "in_compile_signature",
        "in_scientific_config_hash",
    ):
        assert payload[field] is False
    payload["name"] = "changed"
    assert runtime_schema_payload()["name"] == RUNTIME_SCHEMA_NAME
