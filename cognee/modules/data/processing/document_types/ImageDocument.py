from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.chunking.TextChunker import TextChunker
from .Document import Document

class ImageDocument(Document):
    type: str = "image"

    def read(self, chunk_size: int):
        # Transcribe the image file
        result = get_llm_client().transcribe_image(self.raw_data_location)
        text = result.choices[0].message.content

        chunker = TextChunker(self, chunk_size = chunk_size, get_text = lambda: text)

        yield from chunker.read()
