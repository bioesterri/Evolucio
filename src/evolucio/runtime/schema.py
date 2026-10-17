"""Versioned host runtime contract, outside scientific configuration hashes."""

import hashlib
import json

RUNTIME_SCHEMA_NAME = "host_chunked_run_runtime_v1"
RUNTIME_SCHEMA_VERSION = 1


def runtime_schema_payload() -> dict[str, object]:
    """Return a fresh JSON-compatible runtime contract."""
    return {
        "name": RUNTIME_SCHEMA_NAME,
        "version": RUNTIME_SCHEMA_VERSION,
        "kernel": "run_chunk",
        "plan": {"type": "RunPlan", "fields": ["total_steps", "chunk_steps"]},
        "total_steps": "exact_requested_horizon",
        "partition": "divmod_total_steps_chunk_steps_full_chunks_then_remainder",
        "partial_chunk": "run_chunk_with_exact_remainder_without_padding",
        "device_resident": ["SimulationState", "MetricsAccumulator"],
        "host_transfer": "jax.device_get((chunk.metrics, chunk.events))_after_each_chunk",
        "consumer": "synchronous_stream_of_HostChunkBatch",
        "events": "all_births_and_deaths_in_temporal_order_without_sampling",
        "retained_history": False,
        "early_stop": False,
        "persistence": False,
        "chunk_steps": "technical_and_semantically_neutral",
        "in_compile_signature": False,
        "in_scientific_config_hash": False,
    }


def runtime_schema_digest() -> str:
    """Return SHA-256 of the canonical JSON contract."""
    canonical = json.dumps(
        runtime_schema_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


RUNTIME_SCHEMA_DIGEST = runtime_schema_digest()
