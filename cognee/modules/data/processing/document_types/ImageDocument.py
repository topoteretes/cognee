from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.chunking.TextChunker import TextChunker
from .Document import Document

class ImageDocument(Document):
    type: str = "image"


    def transcribe_image(self):
        result = get_llm_client().transcribe_image(self.raw_data_location)
        return(result.choices[0].message.content)

    def read(self, chunk_size: int):
        # Transcribe the image file
        text = self.transcribe_image()

        chunker = TextChunker(self, chunk_size = chunk_size, get_text = lambda: [text])

        yield from chunker.read()
