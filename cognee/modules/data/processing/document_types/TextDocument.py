from .Document import Document
from cognee.modules.chunking.Chunker import Chunker
from .open_data_file import open_data_file


class TextDocument(Document):
    type: str = "text"

    def read(self, chunker_cls: Chunker, max_chunk_size: int):
        def get_text():
            with open_data_file(self.raw_data_location, mode="r", encoding="utf-8") as file:
                while True:
                    text = file.read(1000000)
                    if not text.strip():
                        break
                    yield text

        chunker = chunker_cls(self, max_chunk_size=max_chunk_size, get_text=get_text)
        yield from chunker.read()
