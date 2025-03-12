from pypdf import PdfReader
from cognee.modules.chunking.Chunker import Chunker

from .Document import Document


class PdfDocument(Document):
    type: str = "pdf"

    def read(self, chunker_cls: Chunker, max_chunk_size: int):
        file = PdfReader(self.raw_data_location)

        def get_text():
            for page in file.pages:
                page_text = page.extract_text()
                yield page_text

        chunker = chunker_cls(self, get_text=get_text, max_chunk_size=max_chunk_size)

        yield from chunker.read()

        file.stream.close()
