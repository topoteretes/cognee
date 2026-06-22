from __future__ import annotations

from typing import Any
from uuid import UUID

MAX_SERIALIZED_VALUE_LENGTH = 1000
MAX_TRACE_CONTAINER_ITEMS = 20


def truncate_text(value: str, limit: int) -> str:
    """Bound stored trace strings so unusually large params/returns do not create oversized trace payloads."""
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def sanitize_value(value: Any) -> Any:
    """Make runtime values safe to persist by normalizing custom objects, trimming containers, and keeping JSON serialization reliable."""
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
        sanitized: dict[str, Any] = {}
        for key, item in list(value.items())[:MAX_TRACE_CONTAINER_ITEMS]:
            sanitized[str(key)] = sanitize_value(item)
        return sanitized
    if hasattr(value, "id") and hasattr(value, "__class__"):
        return {
            "type": value.__class__.__name__,
            "id": str(getattr(value, "id", "")),
        }
    return truncate_text(str(value), MAX_SERIALIZED_VALUE_LENGTH)
