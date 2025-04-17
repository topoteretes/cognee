from cognee.modules.data.models import Data
import json
from cognee.modules.data.processing.document_types import (
    Document,
    PdfDocument,
    AudioDocument,
    ImageDocument,
    TextDocument,
    UnstructuredDocument,
)
from cognee.infrastructure.engine import DataPoint
from typing import List, Optional
import uuid

# UUID namespace for consistent ID generation
NAMESPACE_UUID = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # UUID for DNS namespace

EXTENSION_TO_DOCUMENT_CLASS = {
    "pdf": PdfDocument,  # Text documents
    "txt": TextDocument,
    "docx": UnstructuredDocument,
    "doc": UnstructuredDocument,
    "odt": UnstructuredDocument,
    "xls": UnstructuredDocument,
    "xlsx": UnstructuredDocument,
    "ppt": UnstructuredDocument,
    "pptx": UnstructuredDocument,
    "odp": UnstructuredDocument,
    "ods": UnstructuredDocument,
    "png": ImageDocument,  # Image documents
    "dwg": ImageDocument,
    "xcf": ImageDocument,
    "jpg": ImageDocument,
    "jpx": ImageDocument,
    "apng": ImageDocument,
    "gif": ImageDocument,
    "webp": ImageDocument,
    "cr2": ImageDocument,
    "tif": ImageDocument,
    "bmp": ImageDocument,
    "jxr": ImageDocument,
    "psd": ImageDocument,
    "ico": ImageDocument,
    "heic": ImageDocument,
    "avif": ImageDocument,
    "aac": AudioDocument,  # Audio documents
    "mid": AudioDocument,
    "mp3": AudioDocument,
    "m4a": AudioDocument,
    "ogg": AudioDocument,
    "flac": AudioDocument,
    "wav": AudioDocument,
    "amr": AudioDocument,
    "aiff": AudioDocument,
}


class NodeSet(DataPoint):
    """NodeSet data point."""

    name: str
    type: str = "NodeSet"

    metadata: dict = {"index_fields": ["name"]}


def update_node_set(document):
    """Extracts node_set from document's external_metadata."""
    try:
        external_metadata = json.loads(document.external_metadata)
    except json.JSONDecodeError:
        return

    if not isinstance(external_metadata, dict):
        return

    if "node_set" not in external_metadata:
        return

    node_set = external_metadata["node_set"]
    if not isinstance(node_set, list):
        return

    document.node_set = [
        NodeSet(id=uuid.uuid5(NAMESPACE_UUID, f"NodeSet:{node_set_name}"), name=node_set_name)
        for node_set_name in node_set
    ]


async def classify_documents(data_documents: list[Data]) -> list[Document]:
    """
    Classifies a list of data items into specific document types based on file extensions.

    Notes:
        - The function relies on `get_metadata` to retrieve metadata information for each data item.
        - Ensure the `Data` objects and their attributes (e.g., `extension`, `id`) are valid before calling this function.
    """
    documents = []
    for data_item in data_documents:
        document = EXTENSION_TO_DOCUMENT_CLASS[data_item.extension](
            id=data_item.id,
            title=f"{data_item.name}.{data_item.extension}",
            raw_data_location=data_item.raw_data_location,
            name=data_item.name,
            mime_type=data_item.mime_type,
            external_metadata=json.dumps(data_item.external_metadata, indent=4),
        )
        update_node_set(document)
        documents.append(document)

    return documents
