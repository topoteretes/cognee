from typing import Optional, List
from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.Chunker import Chunker


class Document(DataPoint):
    name: str
    raw_data_location: str
    external_metadata: Optional[str]
    mime_type: str
    metadata: dict = {"index_fields": ["name"]}

    async def read(self, chunker_cls: Chunker, max_chunk_size: int) -> str:
        pass
