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
        documents.append(document)

    return documents
