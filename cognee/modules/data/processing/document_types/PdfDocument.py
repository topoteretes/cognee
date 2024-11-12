from pypdf import PdfReader
from cognee.modules.chunking.TextChunker import TextChunker
from .Document import Document

class PdfDocument(Document):
    type: str = "pdf"

    def read(self, chunk_size: int):
        file = PdfReader(self.raw_data_location)

        def get_text():
            for page in file.pages:
                page_text = page.extract_text()
                yield page_text

        chunker = TextChunker(self, chunk_size = chunk_size, get_text = get_text)

        yield from chunker.read()

        file.stream.close()
