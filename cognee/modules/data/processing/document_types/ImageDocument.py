from typing import Optional

from cognee.infrastructure.llm.get_llm_client import get_llm_client

from .ChunkerMapping import ChunkerConfig
from .Document import Document


class ImageDocument(Document):
    type: str = "image"

    def transcribe_image(self):
        result = get_llm_client().transcribe_image(self.raw_data_location)
        return result.choices[0].message.content

    def read(self, chunk_size: int, chunker: str, max_tokens: Optional[int] = None):
        # Transcribe the image file
        text = self.transcribe_image()

        chunker_func = ChunkerConfig.get_chunker(chunker)
        chunker = chunker_func(
            self, chunk_size=chunk_size, get_text=lambda: [text], max_tokens=max_tokens
        )

        yield from chunker.read()
