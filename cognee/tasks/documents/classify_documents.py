from cognee.modules.data.models import Data
from cognee.modules.data.processing.document_types import (
    Document,
    PdfDocument,
    AudioDocument,
    ImageDocument,
    TextDocument,
    UnstructuredDocument,
)
from cognee.modules.data.operations.get_metadata import get_metadata

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
    documents = []
    for data_item in data_documents:
        metadata = await get_metadata(data_item.id)
        document = EXTENSION_TO_DOCUMENT_CLASS[data_item.extension](
            id=data_item.id,
            title=f"{data_item.name}.{data_item.extension}",
            raw_data_location=data_item.raw_data_location,
            name=data_item.name,
            mime_type=data_item.mime_type,
            metadata_id=metadata.id,
        )
        documents.append(document)

    return documents
