from uuid import UUID, uuid5, NAMESPACE_OID
from pypdf import PdfReader
from cognee.modules.data.chunking.DocumentChunker import DocumentChunker
from .Document import Document

class PdfDocument(Document):
    type: str = "pdf"
    title: str
    file_path: str

    def __init__(self, id: UUID, title: str, file_path: str):
        self.id = id or uuid5(NAMESPACE_OID, title)
        self.title = title
        self.file_path = file_path

    def read(self) -> PdfReader:
        file = PdfReader(self.file_path)
        chunker = DocumentChunker(self.id)

        for page in file.pages:
            page_text = page.extract_text()

            for chunk in chunker.read(page_text):
                yield chunk

        file.stream.close()

    def to_dict(self) -> dict:
        return dict(
            id = str(self.id),
            type = self.type,
            title = self.title,
            file_path = self.file_path,
        )
