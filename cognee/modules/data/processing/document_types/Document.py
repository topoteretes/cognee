from typing import List, Optional

from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.Chunker import Chunker


class Document(DataPoint):
    name: str
    raw_data_location: str
    external_metadata: str | None
    mime_type: str
    metadata: dict = {"index_fields": ["name"]}
    importance_weight: float | None = 0.5

    async def read(self, chunker_cls: Chunker, max_chunk_size: int) -> str:
        pass
