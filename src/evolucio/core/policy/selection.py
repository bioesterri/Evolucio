"""Deterministic conversion of policy scores into action proposals."""

import hashlib
import json

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.codes import ACTION_COUNT, ActionCode
from evolucio.core.dtypes import CODE_DTYPE, COUNT_DTYPE, MASK_DTYPE, REAL_DTYPE
from evolucio.core.types import Array

from .schema import policy_schema_digest

ACTION_SELECTION_SCHEMA_NAME = "deterministic_max_score_lowest_action_code_v1"
ACTION_SELECTION_SCHEMA_VERSION = 1
ACTION_SELECTION_TIE_BREAK = "lowest_action_code"
ACTION_SELECTION_INVALID_FALLBACK = "stay"
ACTION_SELECTION_INACTIVE_FALLBACK = "stay"


class PolicyDecisionResult(eqx.Module):
    """Canonical scores, action proposals, and scalar diagnostics."""

    scores: Array
    proposed_actions: Array
    invalid_active_score_count: Array
    exact_tie_count: Array


def action_selection_schema_payload() -> dict[str, object]:
    """Return the complete JSON-compatible deterministic-selection contract."""
    return {
        "name": ACTION_SELECTION_SCHEMA_NAME,
        "version": ACTION_SELECTION_SCHEMA_VERSION,
        "policy_schema_digest": policy_schema_digest(),
        "score_count": ACTION_COUNT,
        "action_codes": [{"name": code.name, "value": int(code)} for code in ActionCode],
        "selection": "maximum_score",
        "exact_tie_definition": "two_or_more_scores_exactly_equal_to_row_maximum",
        "tie_break": ACTION_SELECTION_TIE_BREAK,
        "inactive_fallback": ACTION_SELECTION_INACTIVE_FALLBACK,
        "non_finite_score_fallback": ACTION_SELECTION_INVALID_FALLBACK,
        "uses_output_normalization": False,
        "uses_randomness": False,
        "output_semantics": "proposed_action_not_executed_action",
    }


def action_selection_schema_digest() -> str:
    """Return the canonical SHA-256 digest of the selection schema."""
    canonical = json.dumps(
        action_selection_schema_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


ACTION_SELECTION_SCHEMA_DIGEST = action_selection_schema_digest()


def select_actions_deterministically(raw_scores: Array, alive: Array) -> PolicyDecisionResult:
    """Select the lowest action code among exact finite row maxima."""
    if raw_scores.ndim != 2 or raw_scores.shape[1] != ACTION_COUNT:
        raise ValueError(
            f"raw_scores: expected shape [C,{ACTION_COUNT}], received {raw_scores.shape}"
        )
    if raw_scores.dtype != jnp.dtype(REAL_DTYPE):
        raise TypeError(f"raw_scores: expected dtype float32, received {raw_scores.dtype}")
    if alive.shape != (raw_scores.shape[0],):
        raise ValueError(f"alive: expected shape ({raw_scores.shape[0]},), received {alive.shape}")
    if alive.dtype != jnp.dtype(MASK_DTYPE):
        raise TypeError(f"alive: expected dtype bool, received {alive.dtype}")

    finite_rows = jnp.all(jnp.isfinite(raw_scores), axis=1)
    usable_rows = alive & finite_rows
    zero = jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
        0.0, dtype=REAL_DTYPE
    )
    scores = jnp.where(usable_rows[:, None], raw_scores, zero)
    row_maxima = jnp.max(scores, axis=1, keepdims=True)
    is_maximum = scores == row_maxima
    action_indices = jnp.arange(  # pyright: ignore[reportUnknownMemberType]
        ACTION_COUNT, dtype=CODE_DTYPE
    )
    sentinel = jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
        ACTION_COUNT, dtype=CODE_DTYPE
    )
    candidates = jnp.where(is_maximum, action_indices[None, :], sentinel)
    selected = jnp.min(candidates, axis=1)
    stay = jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
        ActionCode.STAY, dtype=CODE_DTYPE
    )
    proposed_actions = jnp.where(usable_rows, selected, stay).astype(CODE_DTYPE)
    maximum_count = jnp.sum(is_maximum, axis=1, dtype=COUNT_DTYPE)
    exact_ties = usable_rows & (maximum_count > 1)
    invalid_active = alive & ~finite_rows
    return PolicyDecisionResult(
        scores=scores,
        proposed_actions=proposed_actions,
        invalid_active_score_count=jnp.sum(invalid_active, dtype=COUNT_DTYPE),
        exact_tie_count=jnp.sum(exact_ties, dtype=COUNT_DTYPE),
    )
