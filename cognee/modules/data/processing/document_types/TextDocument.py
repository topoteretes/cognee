from cognee.modules.chunking.Chunker import Chunker
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from .Document import Document


class TextDocument(Document):
    type: str = "text"
    mime_type: str = "text/plain"

    async def read(self, chunker_cls: Chunker, max_chunk_size: int):
        async def get_text():
            async with open_data_file(self.raw_data_location, mode="r", encoding="utf-8") as file:
                while True:
                    text = file.read(1000000)
                    if not text.strip():
                        break
                    yield text

        chunker = chunker_cls(self, max_chunk_size=max_chunk_size, get_text=get_text)

        async for chunk in chunker.read():
            yield chunk
