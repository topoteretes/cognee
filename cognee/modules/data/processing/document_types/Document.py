from typing import Optional
from uuid import UUID

from cognee.infrastructure.engine import DataPoint


class Document(DataPoint):
    name: str
    raw_data_location: str
    external_metadata: Optional[str]
    mime_type: str
    token_count: Optional[int] = None
    _metadata: dict = {"index_fields": ["name"], "type": "Document"}

    def read(self, chunk_size: int, chunker=str) -> str:
        pass
