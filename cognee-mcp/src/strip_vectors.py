"""
Filter embedding vectors from cognee search results before returning to MCP clients.

Search results from CHUNKS and SUMMARIES types include `text_vector` fields containing
raw embedding vectors (e.g. 4096 floats, ~92KB per result). These are useless for LLM
clients and quickly exhaust context windows.

This module provides a recursive filter that strips `text_vector` from any result
structure (dict, list, Pydantic model, or plain object) before MCP serialization.
"""

from typing import Any


def strip_vectors(obj: Any) -> Any:
    """Recursively remove text_vector fields from search results."""
    if isinstance(obj, dict):
        return {k: strip_vectors(v) for k, v in obj.items() if k != "text_vector"}
    elif isinstance(obj, list):
        return [strip_vectors(item) for item in obj]
    elif hasattr(obj, "model_dump") and callable(obj.model_dump):
        d = obj.model_dump()
        return {k: strip_vectors(v) for k, v in d.items() if k != "text_vector"}
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        d = vars(obj)
        return {k: strip_vectors(v) for k, v in d.items() if k != "text_vector"}
    return obj
