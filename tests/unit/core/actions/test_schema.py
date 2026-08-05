import re

from evolucio.core.actions import (
    ACTION_CONTRACT_SCHEMA_NAME,
    ACTION_CONTRACT_SCHEMA_VERSION,
    ACTION_INVALID_FALLBACK,
    ACTION_VALIDATION_SCOPE,
    action_contract_schema_digest,
    action_contract_schema_payload,
)


def test_action_contract_payload_is_complete_and_canonical() -> None:
    payload = action_contract_schema_payload()
    assert payload["name"] == ACTION_CONTRACT_SCHEMA_NAME
    assert payload["version"] == ACTION_CONTRACT_SCHEMA_VERSION == 1
    assert payload["validation_scope"] == ACTION_VALIDATION_SCOPE == "local_preliminary"
    assert payload["invalid_fallback"] == ACTION_INVALID_FALLBACK == "stay"
    assert payload["validation_precedence"] == [
        "INACTIVE_SLOT",
        "INVALID_ACTION_CODE",
        "INVALID_ACTOR_POSITION",
        "MOVE_OUT_OF_BOUNDS",
        "EAT_NO_RESOURCE",
        "ACCEPTED",
    ]
    assert payload["consults_occupancy"] is False
    assert payload["applies_costs"] is False
    assert payload["uses_randomness"] is False


def test_action_contract_digest_v1_regression() -> None:
    digest = action_contract_schema_digest()
    assert digest == "85dbbbb9418746b480b119e956a2d4c4297b9b3739034db42b1bba79871890c3"
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
