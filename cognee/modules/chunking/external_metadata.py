"""Document ``external_metadata`` as chunks carry it.

Callers attach a free-form JSON object to each ingested document; the
``Document`` node stores it as JSON text. Every new chunk copies that object
(``DocumentChunk.external_metadata``) so the graph chunk node and the vector
payload carry it, and hybrid retrieval can surface allowlisted keys straight
from a search hit.

The copy stays JSON text rather than a ``dict``: LanceDB, the default vector
store, cannot map a ``dict`` field onto an Arrow type, and the graph adapters
serialise dict properties to JSON strings anyway. One representation across
every backend; readers parse it back with ``parse_external_metadata``.

Metadata never fails ingestion: anything that is not a non-empty JSON object
becomes ``None``.
"""

import json
from functools import lru_cache
from typing import Any


def normalize_external_metadata(value: Any) -> str | None:
    """Canonical JSON text of a non-empty metadata object, or ``None``.

    Accepts the ``dict`` a caller supplied or the JSON text a ``Document`` or a
    stored chunk node carries. Empty, non-object and invalid inputs all
    normalise to ``None``.
    """
    if isinstance(value, str):
        return _normalize_text(value)
    return _normalize_object(value)


@lru_cache(maxsize=256)
def _normalize_text(text: str) -> str | None:
    # Every chunk of a document normalises the same document string; cache by
    # that string so a long document parses its metadata once.
    if not text.strip():
        return None
    try:
        return _normalize_object(json.loads(text))
    except ValueError:
        return None


def _normalize_object(value: Any) -> str | None:
    if not isinstance(value, dict) or not value:
        return None
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return None


def document_external_metadata(document: Any) -> str | None:
    """The metadata every chunk cut from ``document`` should carry."""
    return normalize_external_metadata(getattr(document, "external_metadata", None))


def parse_external_metadata(value: Any) -> dict | None:
    """The stored metadata as a non-empty ``dict``, or ``None``."""
    if isinstance(value, dict):
        return dict(value) or None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) and parsed else None
