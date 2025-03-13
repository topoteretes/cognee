from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.chunking.Chunker import Chunker

from .Document import Document


class AudioDocument(Document):
    type: str = "audio"

    def create_transcript(self):
        result = get_llm_client().create_transcript(self.raw_data_location)
        return result.text

    def read(self, chunker_cls: Chunker, max_chunk_size: int):
        # Transcribe the audio file

        text = self.create_transcript()

        chunker = chunker_cls(self, max_chunk_size=max_chunk_size, get_text=lambda: [text])

        yield from chunker.read()
