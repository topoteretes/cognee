from cognee.modules.data.models import Data
from cognee.modules.data.processing.document_types import Document, PdfDocument, AudioDocument, ImageDocument, TextDocument

EXTENSION_TO_DOCUMENT_CLASS = {
    "pdf": PdfDocument,
    "audio": AudioDocument,
    "image": ImageDocument,
    "txt": TextDocument
}

def classify_documents(data_documents: list[Data]) -> list[Document]:
    documents = [
        EXTENSION_TO_DOCUMENT_CLASS[data_item.extension](id = data_item.id, title=f"{data_item.name}.{data_item.extension}", raw_data_location=data_item.raw_data_location, name=data_item.name)
        for data_item in data_documents
    ]
    return documents
