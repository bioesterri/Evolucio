import json
import re

from evolucio.core.policy import (
    ACTION_SELECTION_INACTIVE_FALLBACK,
    ACTION_SELECTION_INVALID_FALLBACK,
    ACTION_SELECTION_SCHEMA_NAME,
    ACTION_SELECTION_SCHEMA_VERSION,
    ACTION_SELECTION_TIE_BREAK,
    action_selection_schema_digest,
    action_selection_schema_payload,
    policy_schema_digest,
)


def test_selection_schema_payload_and_digest() -> None:
    payload = action_selection_schema_payload()
    digest = action_selection_schema_digest()
    assert ACTION_SELECTION_SCHEMA_NAME == "deterministic_max_score_lowest_action_code_v1"
    assert ACTION_SELECTION_SCHEMA_VERSION == 1
    assert ACTION_SELECTION_TIE_BREAK == "lowest_action_code"
    assert ACTION_SELECTION_INVALID_FALLBACK == ACTION_SELECTION_INACTIVE_FALLBACK == "stay"
    assert payload["policy_schema_digest"] == policy_schema_digest()
    assert [item["value"] for item in payload["action_codes"]] == list(range(7))
    assert payload["uses_randomness"] is False
    assert payload["uses_output_normalization"] is False
    assert payload["output_semantics"] == "proposed_action_not_executed_action"
    json.dumps(payload, allow_nan=False)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert digest == action_selection_schema_digest()
    assert digest == "fb5398375ce760eb4335f353167359555d7518aec47ce257e79c5b8d6056603f"
