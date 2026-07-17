from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

MAX_SERIALIZED_VALUE_LENGTH = 1000
MAX_TRACE_CONTAINER_ITEMS = 20

logger = logging.getLogger(__name__)


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
            str_key = str(key) if not isinstance(key, str) else key
            if str_key in sanitized:
                logger.warning(
                    "sanitize_value: dict key collision after str() conversion — "
                    "key %r collides with an earlier key; appending suffix to avoid "
                    "silent data loss",
                    key,
                )
                suffix = 2
                while f"{str_key}_{suffix}" in sanitized:
                    suffix += 1
                str_key = f"{str_key}_{suffix}"
            sanitized[str_key] = sanitize_value(item)
        return sanitized
    if hasattr(value, "id") and hasattr(value, "__class__"):
        return {
            "type": value.__class__.__name__,
            "id": str(getattr(value, "id", "")),
        }
    return truncate_text(str(value), MAX_SERIALIZED_VALUE_LENGTH)
