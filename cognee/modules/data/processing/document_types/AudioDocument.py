from cognee.infrastructure.llm.get_llm_client import get_llm_client

from .Document import Document


class AudioDocument(Document):
    type: str = "audio"

    def create_transcript(self):
        result = get_llm_client().create_transcript(self.raw_data_location)
        return result.text

    def read(self, chunk_size: int, chunker_cls: type, max_chunk_tokens: int):
        # Transcribe the audio file

        text = self.create_transcript()

        chunker = chunker_cls(
            self, chunk_size=chunk_size, get_text=lambda: [text], max_chunk_tokens=max_chunk_tokens
        )

        yield from chunker.read()
