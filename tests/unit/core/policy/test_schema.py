import json
import re
from dataclasses import asdict

from evolucio.core.codes import ACTION_COUNT, ActionCode
from evolucio.core.observations import OBSERVATION_SIZE, observation_schema_digest
from evolucio.core.policy import (
    POLICY_ACTIVATION_NAME,
    POLICY_HIDDEN_SIZE,
    POLICY_INPUT_SIZE,
    POLICY_OUTPUT_SIZE,
    POLICY_PARAMETER_COUNT,
    POLICY_PARAMETER_SPECS,
    POLICY_SCHEMA_DIGEST,
    POLICY_SCHEMA_NAME,
    POLICY_SCHEMA_VERSION,
    POLICY_USE_BIAS,
    policy_schema_digest,
    policy_schema_payload,
)


def test_fixed_schema_constants_and_parameter_order() -> None:
    assert (POLICY_INPUT_SIZE, POLICY_HIDDEN_SIZE, POLICY_OUTPUT_SIZE) == (15, 16, 7)
    assert POLICY_INPUT_SIZE == OBSERVATION_SIZE
    assert POLICY_OUTPUT_SIZE == ACTION_COUNT
    assert POLICY_PARAMETER_COUNT == 16 * 15 + 16 + 7 * 16 + 7 == 375
    assert POLICY_ACTIVATION_NAME == "tanh" and POLICY_USE_BIAS is True
    assert [(s.path, s.shape, s.count) for s in POLICY_PARAMETER_SPECS] == [
        ("layer1.weight", (16, 15), 240),
        ("layer1.bias", (16,), 16),
        ("layer2.weight", (7, 16), 112),
        ("layer2.bias", (7,), 7),
    ]
    assert len({s.path for s in POLICY_PARAMETER_SPECS}) == 4
    assert all(s.dtype == "float32" for s in POLICY_PARAMETER_SPECS)
    assert all(
        not hasattr(value, "dtype") for s in POLICY_PARAMETER_SPECS for value in asdict(s).values()
    )


def test_payload_binds_observations_actions_and_linear_scores() -> None:
    payload = policy_schema_payload()
    assert json.loads(json.dumps(payload))
    assert payload["observation_schema"]["digest"] == observation_schema_digest()  # type: ignore[index]
    assert payload["action_codes"] == [{"name": c.name, "value": int(c)} for c in ActionCode]
    assert payload["output_activation"] is None
    assert payload["output_semantics"] == "linear_action_scores"


def test_v1_digest_regression() -> None:
    assert POLICY_SCHEMA_NAME == "policy_mlp_tanh_15_16_7_v1"
    assert POLICY_SCHEMA_VERSION == 1
    assert (
        POLICY_SCHEMA_DIGEST
        == policy_schema_digest()
        == "b1c8336c8c0be45f3bacbbf384ccb5d8cbf352651c084da3ac98fcab6bbc7e90"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", POLICY_SCHEMA_DIGEST)
