"""Shared utilities for DLT ingestion."""

import json
from typing import Optional

from cognee.shared.logging_utils import get_logger

logger = get_logger("dlt_utils")

# A dlt source sets this attribute to opt into the "document" ingestion path:
# each row becomes a text document that flows through normal cognify (LLM entity
# extraction), instead of the default relational schema-context path. The value
# is the ``external_metadata["source"]`` tag used for that source's rows. This
# lets resolve_dlt_sources stay connector-agnostic — a connector declares its
# own nature rather than the shared engine hard-coding connector names.
DOCUMENT_SOURCE_ATTR = "cognee_document_source"


def document_source_tag(item) -> Optional[str]:
    """Return the document-source tag a dlt source opted into, else ``None``."""
    tag = getattr(item, DOCUMENT_SOURCE_ATTR, None)
    return tag if isinstance(tag, str) and tag else None


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
            # There WAS metadata but it could not be decoded — surface it so
            # the lost FK edges are diagnosable rather than silently dropped.
            logger.warning(
                "Failed to parse external_metadata as JSON for object id=%s; "
                "treating as absent. Any DLT FK edges for this row are skipped.",
                getattr(obj, "id", "<unknown>"),
            )
            return None
    return None
