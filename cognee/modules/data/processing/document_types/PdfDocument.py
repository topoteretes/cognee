from pypdf import PdfReader
from cognee.modules.chunking.Chunker import Chunker

from .Document import Document


class PdfDocument(Document):
    type: str = "pdf"

    def read(self, chunk_size: int, chunker_cls: Chunker, max_chunk_tokens: int):
        file = PdfReader(self.raw_data_location)

        def get_text():
            for page in file.pages:
                page_text = page.extract_text()
                yield page_text

        chunker = chunker_cls(
            self, chunk_size=chunk_size, get_text=get_text, max_chunk_tokens=max_chunk_tokens
        )

        yield from chunker.read()

        file.stream.close()
