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
    assert INVARIANT_SCHEMA_NAME == "core_state_transition_invariants_v1"
    assert INVARIANT_SCHEMA_VERSION == 1
    assert invariant_schema_payload()["policy"] == "detect, never repair"
    assert invariant_schema_digest() == INVARIANT_SCHEMA_DIGEST
    assert INVARIANT_SCHEMA_DIGEST == (
        "3864d23237de0273da59d627e503007820a1c4ada42dc99dd70cb79485bd6d1e"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", INVARIANT_SCHEMA_DIGEST)
