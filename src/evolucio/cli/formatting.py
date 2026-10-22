"""Deterministic terminal formatting for JSON-compatible host values."""

import json
from collections.abc import Mapping


def json_text(value: object) -> str:
    """Render canonical key ordering and unambiguous finite scalar values."""
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)


def text_fields(values: Mapping[str, object]) -> str:
    """Render named values without Python or device object representations."""
    return "\n".join(
        f"{name}: {json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)}"
        for name, value in values.items()
    )
