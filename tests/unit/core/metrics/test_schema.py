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
    assert METRICS_SCHEMA_NAME == "step_metrics_events_accumulators_v2"
    assert METRICS_SCHEMA_VERSION == 2
    assert payload["fitness"] is False
    assert payload["persistence"] is False
    assert payload["temporal_semantics"]["alive_count"] == "final_state"
    assert payload["temporal_semantics"]["birth_count"] == "within_step_flow"
    assert payload["death_cause_codes"] == [
        {"name": "NONE", "value": 0},
        {"name": "ENERGY_DEPLETION", "value": 1},
        {"name": "MAX_AGE", "value": 2},
        {"name": "ENVIRONMENTAL_STRESS", "value": 3},
        {"name": "COMPETITIVE_EXCLUSION", "value": 4},
        {"name": "INVALID_STATE", "value": 5},
    ]
    assert metrics_schema_digest() == METRICS_SCHEMA_DIGEST
    assert METRICS_SCHEMA_DIGEST == (
        "d7d337952ee03acbc2824a1994223f3bf6753b1cb064c68e5f3bbeb9212f318d"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", METRICS_SCHEMA_DIGEST)
