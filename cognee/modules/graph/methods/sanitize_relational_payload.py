from typing import Any


def sanitize_relational_payload(value: Any) -> Any:
    """Normalize payload values before relational writes.

    Removes Postgres-incompatible NUL bytes from strings and recursively applies
    the same cleanup to nested container values. Byte sequences are decoded with
    replacement so invalid UTF-8 does not break persistence.
    """
    if isinstance(value, str):
        return value.replace("\x00", "")

    if isinstance(value, (bytes, bytearray)):
        return sanitize_relational_payload(bytes(value).decode("utf-8", errors="replace"))

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
