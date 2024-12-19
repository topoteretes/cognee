from pypdf import PdfReader
from .Document import Document
from .ChunkerMapping import ChunkerConfig

class PdfDocument(Document):
    type: str = "pdf"

    def read(self, chunk_size: int, chunker: str):
        file = PdfReader(self.raw_data_location)

        def get_text():
            for page in file.pages:
                page_text = page.extract_text()
                yield page_text

        chunker_func = ChunkerConfig.get_chunker(chunker)
        chunker = chunker_func(self, chunk_size = chunk_size, get_text = get_text)

        yield from chunker.read()

        file.stream.close()
