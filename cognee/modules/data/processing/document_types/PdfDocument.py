from uuid import UUID, uuid5, NAMESPACE_OID
from pypdf import PdfReader
from cognee.modules.data.chunking.TextChunker import TextChunker
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

        def get_text():
            for page in file.pages:
                page_text = page.extract_text()
                yield page_text

        chunker = TextChunker(self.id, get_text = get_text)

        yield from chunker.read()

        file.stream.close()

    def to_dict(self) -> dict:
        return dict(
            id = str(self.id),
            type = self.type,
            title = self.title,
            file_path = self.file_path,
        )
