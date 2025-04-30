from pypdf import PdfReader
from cognee.modules.chunking.Chunker import Chunker
from .open_data_file import open_data_file
from .Document import Document


class PdfDocument(Document):
    type: str = "pdf"

    def read(self, chunker_cls: Chunker, max_chunk_size: int):
        with open_data_file(self.raw_data_location, mode="rb") as stream:
            file = PdfReader(stream)

            def get_text():
                for page in file.pages:
                    page_text = page.extract_text()
                    yield page_text

            chunker = chunker_cls(self, get_text=get_text, max_chunk_size=max_chunk_size)

            yield from chunker.read()
