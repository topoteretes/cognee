from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.chunking.TextChunker import TextChunker
from .Document import Document

class AudioDocument(Document):
    type: str = "audio"

    def read(self, chunk_size: int):
        # Transcribe the audio file
        result = get_llm_client().create_transcript(self.raw_data_location)
        text = result.text

        chunker = TextChunker(self, chunk_size = chunk_size, get_text = lambda: text)

        yield from chunker.read()
