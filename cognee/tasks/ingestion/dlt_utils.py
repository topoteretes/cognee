"""Shared utilities for DLT ingestion."""

import json
from typing import Optional

# A dlt source sets this attribute to opt into the "document" ingestion path:
# each row becomes a text document that flows through normal cognify (LLM entity
# extraction), instead of the default relational schema-context path. The value
# is the ``system_metadata["source"]`` tag used for that source's rows. This
# lets resolve_dlt_sources stay connector-agnostic — a connector declares its
# own nature rather than the shared engine hard-coding connector names.
DOCUMENT_SOURCE_ATTR = "cognee_document_source"


def document_source_tag(item) -> Optional[str]:
    """Return the document-source tag a dlt source opted into, else ``None``."""
    tag = getattr(item, DOCUMENT_SOURCE_ATTR, None)
    return tag if isinstance(tag, str) and tag else None


def _metadata_source(metadata) -> Optional[str]:
    """Extract the ``source`` field from system metadata.

    Accepts a dict, a JSON string, or an object with a ``system_metadata``
    attribute (a Data record / DataItem). Returns None when the source cannot
    be determined. Deliberately never reads external_metadata: that field is
    user-writable, and routing/deletion decisions must not key on user bytes.
    """
    meta = getattr(metadata, "system_metadata", metadata)
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(meta, dict):
        return meta.get("source")
    return None


def is_dlt_sourced(metadata) -> bool:
    """Check whether system_metadata indicates a legacy per-row DLT item (source == "dlt")."""
    return _metadata_source(metadata) == "dlt"


def is_dlt_source_manifest(metadata) -> bool:
    """Check whether system_metadata indicates a DLT source manifest (source == "dlt_source")."""
    return _metadata_source(metadata) == "dlt_source"


async def load_dlt_manifest(raw_data_location: str) -> dict:
    """Load a DLT source manifest from storage.

    Single reader of the DLT source manifest format written by
    ``resolve_dlt_sources._build_source_manifest_item``.
    """
    from cognee.infrastructure.files.utils.open_data_file import open_data_file

    async with open_data_file(raw_data_location, mode="r", encoding="utf-8") as file:
        return json.loads(file.read())
