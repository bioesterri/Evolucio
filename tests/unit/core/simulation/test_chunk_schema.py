from evolucio.core.simulation import (
    RUN_CHUNK_EARLY_STOP,
    RUN_CHUNK_OUTPUT_POLICY,
    RUN_CHUNK_SCAN_KIND,
    RUN_CHUNK_SCHEMA_DIGEST,
    RUN_CHUNK_SCHEMA_NAME,
    RUN_CHUNK_SCHEMA_VERSION,
    RUN_CHUNK_UNROLL,
    run_chunk_schema_digest,
    run_chunk_schema_payload,
)


def test_chunk_schema_freezes_scan_and_output_contract() -> None:
    payload = run_chunk_schema_payload()
    assert RUN_CHUNK_SCHEMA_NAME == "compiled_lax_scan_simulation_chunk_v1"
    assert RUN_CHUNK_SCHEMA_VERSION == 1
    assert RUN_CHUNK_SCAN_KIND == "lax_scan"
    assert RUN_CHUNK_UNROLL == 1
    assert RUN_CHUNK_EARLY_STOP is False
    assert RUN_CHUNK_OUTPUT_POLICY == "metrics_and_events_only"
    assert payload["carry"] == ["SimulationState", "MetricsAccumulator"]
    assert payload["outputs"] == ["StepMetrics", "StepEventBuffer"]
    assert payload["state_trajectory"] is False
    assert payload["fast_path_diagnostics"] is False
    assert run_chunk_schema_digest() == RUN_CHUNK_SCHEMA_DIGEST
    assert RUN_CHUNK_SCHEMA_DIGEST == (
        "f448c61cc4de4415d9bb865c54d5588380e87c59f7e11ad240840a5ef7cdf4d9"
    )
