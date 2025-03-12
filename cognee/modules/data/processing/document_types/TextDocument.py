from .Document import Document
from cognee.modules.chunking.Chunker import Chunker


class TextDocument(Document):
    type: str = "text"

    def read(self, chunker_cls: Chunker, max_chunk_size: int):
        def get_text():
            with open(self.raw_data_location, mode="r", encoding="utf-8") as file:
                while True:
                    text = file.read(1000000)

                    if len(text.strip()) == 0:
                        break

                    yield text

        chunker = chunker_cls(self, max_chunk_size=max_chunk_size, get_text=get_text)

        yield from chunker.read()
