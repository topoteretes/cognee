from cognee.modules.data.models import Data
from cognee.modules.data.processing.document_types import (
    Document,
    PdfDocument,
    AudioDocument,
    ImageDocument,
    TextDocument,
)

EXTENSION_TO_DOCUMENT_CLASS = {
    "pdf": PdfDocument,  # Text documents
    "txt": TextDocument,
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


def classify_documents(data_documents: list[Data]) -> list[Document]:
    documents = [
        EXTENSION_TO_DOCUMENT_CLASS[data_item.extension](
            id=data_item.id,
            title=f"{data_item.name}.{data_item.extension}",
            raw_data_location=data_item.raw_data_location,
            name=data_item.name,
        )
        for data_item in data_documents
    ]
    return documents
