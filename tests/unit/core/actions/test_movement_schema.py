import json

from evolucio.core.actions import (
    ACTION_CONTRACT_SCHEMA_DIGEST,
    MOVEMENT_RESOLUTION_CODE_COUNT,
    MOVEMENT_RESOLUTION_SCHEMA_DIGEST,
    MOVEMENT_RESOLUTION_SCHEMA_NAME,
    MOVEMENT_RESOLUTION_SCHEMA_VERSION,
    MovementConflictPriorityStreamCode,
    MovementResolutionCode,
    movement_resolution_schema_payload,
)


def test_movement_codes_substreams_and_schema_are_frozen() -> None:
    assert [(code.name, code.value) for code in MovementResolutionCode] == [
        ("NOT_MOVEMENT", 0),
        ("MOVED", 1),
        ("DESTINATION_OCCUPIED", 2),
        ("CONFLICT_LOST", 3),
        ("PRIORITY_COLLISION", 4),
        ("INVALID_MOVEMENT_INPUT", 5),
    ]
    assert MOVEMENT_RESOLUTION_CODE_COUNT == 6
    assert [int(code) for code in MovementConflictPriorityStreamCode] == [0, 1, 2]
    payload = movement_resolution_schema_payload()
    assert payload["name"] == MOVEMENT_RESOLUTION_SCHEMA_NAME
    assert MOVEMENT_RESOLUTION_SCHEMA_VERSION == payload["version"] == 1
    assert payload["action_contract_digest"] == ACTION_CONTRACT_SCHEMA_DIGEST
    assert json.loads(json.dumps(payload)) == payload
    assert MOVEMENT_RESOLUTION_SCHEMA_DIGEST == (
        "9209b617b2ed80ae1f1fa90206f13d05a1c69c763ece24ef92be6f64959a2e03"
    )
