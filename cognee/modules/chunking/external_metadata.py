"""Document ``external_metadata`` as chunks carry it.

Callers attach a free-form JSON object to each ingested document; the
``Document`` node stores it as JSON text. Every new chunk copies that object
(``DocumentChunk.external_metadata``) so the graph chunk node and the vector
payload carry it, and hybrid retrieval can surface allowlisted keys straight
from a search hit.

The copy stays JSON text rather than a ``dict``: LanceDB, the default vector
store, cannot map a ``dict`` field onto an Arrow type, and the graph adapters
serialise dict properties to JSON strings anyway. One representation across
every backend; readers parse it back with ``decode_external_metadata``.

Metadata never fails ingestion: anything that is not a non-empty JSON object
becomes ``None``.
"""

import json
from typing import Any


def normalize_external_metadata(value: Any) -> str | None:
    """Canonical JSON text of a non-empty metadata object, or ``None``.

    Accepts the ``dict`` a caller supplied or the JSON text a ``Document`` or a
    stored chunk node carries. Empty, non-object and invalid inputs all
    normalise to ``None``.
    """
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if not isinstance(value, dict) or not value:
        return None
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return None


def decode_external_metadata(value: Any) -> dict | None:
    """The stored metadata as a non-empty ``dict``, or ``None``.

    Named apart from ``tasks.ingestion.data_item.parse_external_metadata``,
    which parses the ingest form field (a JSON *array*, raising on bad input)
    rather than one stored object.
    """
    if isinstance(value, dict):
        return dict(value) or None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) and parsed else None
