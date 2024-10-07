from uuid import UUID, uuid5, NAMESPACE_OID
from pypdf import PdfReader
from cognee.modules.chunking.TextChunker import TextChunker
from .Document import Document

class PdfDocument(Document):
    type: str = "pdf"
    title: str
    raw_data_location: str

    def __init__(self, id: UUID, title: str, raw_data_location: str):
        self.id = id or uuid5(NAMESPACE_OID, title)
        self.title = title
        self.raw_data_location = raw_data_location

    def read(self, chunk_size: int) -> PdfReader:
        file = PdfReader(self.raw_data_location)

        def get_text():
            for page in file.pages:
                page_text = page.extract_text()
                yield page_text

        chunker = TextChunker(self.id, chunk_size = chunk_size, get_text = get_text)

        yield from chunker.read()

        file.stream.close()

    def to_dict(self) -> dict:
        return dict(
            id = str(self.id),
            type = self.type,
            title = self.title,
            raw_data_location = self.raw_data_location,
        )
