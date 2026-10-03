import re

from evolucio.core.diagnostics import (
    INVARIANT_SCHEMA_DIGEST,
    INVARIANT_SCHEMA_NAME,
    INVARIANT_SCHEMA_VERSION,
    STATE_INVARIANT_CODE_COUNT,
    TRANSITION_INVARIANT_CODE_COUNT,
    StateInvariantCode,
    TransitionInvariantCode,
    invariant_schema_digest,
    invariant_schema_payload,
)


def test_codes_and_schema_are_frozen() -> None:
    assert [int(code) for code in StateInvariantCode] == list(range(19))
    assert [int(code) for code in TransitionInvariantCode] == list(range(16))
    assert STATE_INVARIANT_CODE_COUNT == 19
    assert TRANSITION_INVARIANT_CODE_COUNT == 16
    assert INVARIANT_SCHEMA_NAME == "core_state_transition_invariants_v2"
    assert INVARIANT_SCHEMA_VERSION == 2
    assert invariant_schema_payload()["policy"] == "detect, never repair"
    assert invariant_schema_digest() == INVARIANT_SCHEMA_DIGEST
    assert INVARIANT_SCHEMA_DIGEST == (
        "732d25002a04698e0aedbb629b978d781573dea7a8245261133f7c20f43869b6"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", INVARIANT_SCHEMA_DIGEST)
