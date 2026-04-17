from typing import Any


def sanitize_relational_payload(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\x00", "")

    if isinstance(value, list):
        return [sanitize_relational_payload(item) for item in value]

    if isinstance(value, tuple):
        return tuple(sanitize_relational_payload(item) for item in value)

    if isinstance(value, dict):
        return {
            sanitize_relational_payload(key): sanitize_relational_payload(item)
            for key, item in value.items()
        }

    return value
