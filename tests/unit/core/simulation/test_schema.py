from evolucio.core.simulation import (
    SIMULATION_STEP_PHASES,
    SIMULATION_STEP_SCHEMA_DIGEST,
    SIMULATION_STEP_SCHEMA_NAME,
    SIMULATION_STEP_SCHEMA_VERSION,
    simulation_step_schema_digest,
    simulation_step_schema_payload,
)


def test_complete_step_schema_is_canonical_and_explicit() -> None:
    payload = simulation_step_schema_payload()
    assert SIMULATION_STEP_SCHEMA_NAME == "full_ecological_evolution_step_v1"
    assert SIMULATION_STEP_SCHEMA_VERSION == 1
    assert payload["phases"] == list(SIMULATION_STEP_PHASES)
    assert payload["rng_advance"] == "exactly_once_unconditionally"
    assert payload["rng_streams"] == [
        "MOVEMENT_CONFLICT",
        "REPRODUCTION_CONFLICT",
        "BIRTH_PLACEMENT",
        "GENOME_MUTATION",
    ]
    assert payload["newborns_act_next_step"] is True
    assert payload["persistence"] is False
    assert simulation_step_schema_digest() == SIMULATION_STEP_SCHEMA_DIGEST
    assert len(SIMULATION_STEP_SCHEMA_DIGEST) == 64
