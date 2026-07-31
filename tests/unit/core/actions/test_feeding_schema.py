import re

from evolucio.core.actions import (
    FEEDING_RESOLUTION_CODE_COUNT,
    FEEDING_RESOLUTION_SCHEMA_DIGEST,
    FeedingResolutionCode,
    feeding_resolution_schema_digest,
    feeding_resolution_schema_payload,
)


def test_feeding_schema_is_complete_and_stable() -> None:
    payload = feeding_resolution_schema_payload()
    assert FEEDING_RESOLUTION_CODE_COUNT == len(FeedingResolutionCode) == 6
    assert payload["allocation_policy"] == "proportional_feasible_demand"
    assert payload["agent_to_agent_transfer"] is False
    assert payload["applies_costs"] is False
    assert feeding_resolution_schema_digest() == FEEDING_RESOLUTION_SCHEMA_DIGEST
    assert FEEDING_RESOLUTION_SCHEMA_DIGEST == (
        "7901e171683053b6c999ea4a932fe745a5ca9940e48dcba3a07c285026cb4d29"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", FEEDING_RESOLUTION_SCHEMA_DIGEST)
