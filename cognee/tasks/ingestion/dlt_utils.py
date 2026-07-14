"""Shared utilities for DLT ingestion."""

import json
from typing import Optional

from cognee.shared.logging_utils import get_logger

logger = get_logger("dlt_utils")


def _metadata_source(metadata) -> Optional[str]:
    """Extract the ``source`` field from external metadata.

    Accepts a dict, a JSON string, or an object with an ``external_metadata``
    attribute. Returns None when the source cannot be determined.
    """
    ext = getattr(metadata, "external_metadata", metadata)
    if isinstance(ext, str):
        try:
            ext = json.loads(ext)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(ext, dict):
        return ext.get("source")
    return None


def is_dlt_sourced(metadata) -> bool:
    """Check whether external_metadata indicates a legacy per-row DLT item (source == "dlt")."""
    return _metadata_source(metadata) == "dlt"


def is_dlt_source_manifest(metadata) -> bool:
    """Check whether external_metadata indicates a DLT source manifest (source == "dlt_source")."""
    return _metadata_source(metadata) == "dlt_source"


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


async def load_dlt_manifest(raw_data_location: str) -> dict:
    """Load a DLT source manifest from storage.

    Single reader of the DLT source manifest format written by
    ``resolve_dlt_sources._build_source_manifest_item``.
    """
    from cognee.infrastructure.files.utils.open_data_file import open_data_file

    async with open_data_file(raw_data_location, mode="r", encoding="utf-8") as file:
        return json.loads(file.read())
