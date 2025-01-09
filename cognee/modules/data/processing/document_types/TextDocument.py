from typing import Optional

from .ChunkerMapping import ChunkerConfig
from .Document import Document


class TextDocument(Document):
    type: str = "text"

    def read(self, chunk_size: int, chunker: str, max_tokens: Optional[int] = None):
        def get_text():
            with open(self.raw_data_location, mode="r", encoding="utf-8") as file:
                while True:
                    text = file.read(1024)

                    if len(text.strip()) == 0:
                        break

                    yield text

        chunker_func = ChunkerConfig.get_chunker(chunker)

        chunker = chunker_func(
            self, chunk_size=chunk_size, get_text=get_text, max_tokens=max_tokens
        )

        yield from chunker.read()
