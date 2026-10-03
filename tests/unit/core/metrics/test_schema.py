import re

from evolucio.core.metrics import (
    METRICS_SCHEMA_DIGEST,
    METRICS_SCHEMA_NAME,
    METRICS_SCHEMA_VERSION,
    metrics_schema_digest,
    metrics_schema_payload,
)


def test_metrics_schema_is_complete_and_frozen() -> None:
    payload = metrics_schema_payload()
    assert METRICS_SCHEMA_NAME == "step_metrics_events_accumulators_v1"
    assert METRICS_SCHEMA_VERSION == 1
    assert payload["fitness"] is False
    assert payload["persistence"] is False
    assert payload["temporal_semantics"]["alive_count"] == "final_state"
    assert payload["temporal_semantics"]["birth_count"] == "within_step_flow"
    assert metrics_schema_digest() == METRICS_SCHEMA_DIGEST
    assert METRICS_SCHEMA_DIGEST == (
        "7c2854a5fb432b41380187f5955aeea7fa86c9a90d8e0907038cffd5ffeb130e"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", METRICS_SCHEMA_DIGEST)
