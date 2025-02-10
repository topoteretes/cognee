from typing import Optional
from cognee.infrastructure.engine import DataPoint


class Document(DataPoint):
    name: str
    raw_data_location: str
    external_metadata: Optional[str]
    mime_type: str
    metadata: dict = {"index_fields": ["name"]}

    def read(self, chunk_size: int, chunker=str, max_chunk_tokens: Optional[int] = None) -> str:
        pass
