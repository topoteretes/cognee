from cognee.modules.data.models import Data
from cognee.modules.data.processing.document_types import Document, PdfDocument, AudioDocument, ImageDocument, TextDocument

def classify_documents(data_documents: list[Data]) -> list[Document]:
    documents = [
        PdfDocument(id = data_item.id, title=f"{data_item.name}.{data_item.extension}", raw_data_location=data_item.raw_data_location) if data_item.extension == "pdf" else
        AudioDocument(id = data_item.id, title=f"{data_item.name}.{data_item.extension}", raw_data_location=data_item.raw_data_location) if data_item.extension == "audio" else
        ImageDocument(id = data_item.id, title=f"{data_item.name}.{data_item.extension}", raw_data_location=data_item.raw_data_location) if data_item.extension == "image" else
        TextDocument(id = data_item.id, title=f"{data_item.name}.{data_item.extension}", raw_data_location=data_item.raw_data_location)
        for data_item in data_documents
    ]

    return documents
