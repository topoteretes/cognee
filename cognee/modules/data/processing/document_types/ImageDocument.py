from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.chunking.Chunker import Chunker

from .Document import Document


class ImageDocument(Document):
    type: str = "image"

    def transcribe_image(self):
        result = get_llm_client().transcribe_image(self.raw_data_location)
        return result.choices[0].message.content

    def read(self, chunker_cls: Chunker, max_chunk_size: int):
        # Transcribe the image file
        text = self.transcribe_image()

        chunker = chunker_cls(self, get_text=lambda: [text], max_chunk_size=max_chunk_size)

        yield from chunker.read()
