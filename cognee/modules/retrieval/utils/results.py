"""Read common fields from retrieval results and plain payload dictionaries."""

from typing import Any
from uuid import UUID


def payload(result: Any) -> dict:
    if isinstance(result, dict):
        return result
    result_payload = getattr(result, "payload", None)
    return result_payload if isinstance(result_payload, dict) else {}


def display_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, UUID)):
        text = str(value).strip()
        return text or None
    return None


def result_id(result: Any) -> str | None:
    result_payload = payload(result)
    return display_value(result_payload.get("id")) or display_value(getattr(result, "id", None))


def first_display_value(*values: Any) -> str | None:
    for value in values:
        text = display_value(value)
        if text:
            return text
    return None
