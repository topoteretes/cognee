from __future__ import annotations

from itertools import islice
from typing import Any
from uuid import UUID

from cognee.shared.dict_keys import unique_string_keys

MAX_SERIALIZED_VALUE_LENGTH = 1000
MAX_TRACE_CONTAINER_ITEMS = 20


def truncate_text(value: str, limit: int) -> str:
    """Bound stored trace strings so unusually large params/returns do not create oversized trace payloads."""
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _key_name(key: Any) -> str:
    """Deterministic string form of a dict key, capped like any stored string."""
    if isinstance(key, str):
        name = key
    elif hasattr(key, "id"):
        name = f"{type(key).__name__}:{key.id}"
    elif type(key).__str__ is object.__str__ and type(key).__repr__ is object.__repr__:
        # The default repr embeds a memory address, which differs on every run.
        name = f"<{type(key).__name__}>"
    else:
        name = str(key)
    return truncate_text(name, MAX_SERIALIZED_VALUE_LENGTH)


def sanitize_value(value: Any) -> Any:
    """Make runtime values safe to persist by normalizing custom objects, trimming containers, and keeping JSON serialization reliable.

    Dict keys become strings. When two keys convert to the same string (``1`` and
    ``"1"``), the key that already was a string keeps its name and the other gets
    a ``_2``, ``_3``, ... suffix, so no value is dropped.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, str):
        return truncate_text(value, MAX_SERIALIZED_VALUE_LENGTH)
    if isinstance(value, list):
        return [sanitize_value(item) for item in value[:MAX_TRACE_CONTAINER_ITEMS]]
    if isinstance(value, tuple):
        return [sanitize_value(item) for item in value[:MAX_TRACE_CONTAINER_ITEMS]]
    if isinstance(value, dict):
        items = list(islice(value.items(), MAX_TRACE_CONTAINER_ITEMS))
        keys = unique_string_keys(
            [_key_name(key) for key, _ in items], [isinstance(key, str) for key, _ in items]
        )
        return {key: sanitize_value(item) for key, (_, item) in zip(keys, items, strict=True)}
    if hasattr(value, "id") and hasattr(value, "__class__"):
        return {
            "type": value.__class__.__name__,
            "id": str(getattr(value, "id", "")),
        }
    return truncate_text(str(value), MAX_SERIALIZED_VALUE_LENGTH)
