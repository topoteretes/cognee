from cognee.modules.chunking.Chunker import Chunker
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from .Document import Document


class JsonDocument(Document):
    """Document type for JSON files.

    Reads the full JSON file content and passes it to the configured chunker
    (typically ``JsonListChunker``).  Works with both flat top-level JSON arrays
    and nested JSON structures containing arrays.
    """

    type: str = "json"
    mime_type: str = "application/json"

    async def read(self, chunker_cls: Chunker, max_chunk_size: int):
        async def get_text():
            async with open_data_file(
                self.raw_data_location, mode="r", encoding="utf-8"
            ) as file:
                content = file.read()
                if not content.strip():
                    return
                yield content

        chunker = chunker_cls(self, max_chunk_size=max_chunk_size, get_text=get_text)

        async for chunk in chunker.read():
            yield chunk
