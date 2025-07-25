from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.chunking.Chunker import Chunker

from .Document import Document


class AudioDocument(Document):
    type: str = "audio"

    async def create_transcript(self):
        result = await get_llm_client().create_transcript(self.raw_data_location)
        return result.text

    async def read(self, chunker_cls: Chunker, max_chunk_size: int):
        async def get_text():
            # Transcribe the audio file
            yield await self.create_transcript()

        chunker = chunker_cls(self, max_chunk_size=max_chunk_size, get_text=get_text)

        async for chunk in chunker.read():
            yield chunk
