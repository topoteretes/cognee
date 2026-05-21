"""Shared utilities for DLT ingestion."""

import json


def is_dlt_sourced(metadata) -> bool:
    """Check whether external_metadata indicates a DLT source.

    Accepts a dict, a JSON string, or an object with an ``external_metadata``
    attribute.  Returns True when ``source == "dlt"``.
    """
    ext = getattr(metadata, "external_metadata", metadata)
    if isinstance(ext, dict):
        return ext.get("source") == "dlt"
    if isinstance(ext, str):
        try:
            return json.loads(ext).get("source") == "dlt"
        except (json.JSONDecodeError, TypeError):
            pass
    return False


def parse_external_metadata(obj) -> dict | None:
    """Parse external_metadata from an object (may be JSON string or dict).

    Returns the parsed dict or None if parsing fails / not present.
    """
    raw = getattr(obj, "external_metadata", None)
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    return None
