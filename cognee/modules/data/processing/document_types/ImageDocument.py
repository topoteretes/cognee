from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.chunking.Chunker import Chunker

from .Document import Document


class ImageDocument(Document):
    type: str = "image"

    async def transcribe_image(self):
        result = await get_llm_client().transcribe_image(self.raw_data_location)
        return result.choices[0].message.content

    async def read(self, chunker_cls: Chunker, max_chunk_size: int):
        async def get_text():
            # Transcribe the image file
            yield await self.transcribe_image()

        chunker = chunker_cls(self, get_text=get_text, max_chunk_size=max_chunk_size)

        async for chunk in chunker.read():
            yield chunk
