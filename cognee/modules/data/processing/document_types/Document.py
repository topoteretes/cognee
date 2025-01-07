from typing import Optional
from uuid import UUID

from cognee.infrastructure.engine import DataPoint


class Document(DataPoint):
    name: str
    raw_data_location: str
    metadata_id: UUID
    mime_type: str
    _metadata: dict = {
        "index_fields": ["name"],
        "type": "Document"
    }

    def read(self, chunk_size: int, embedding_model: Optional[str], max_tokens: Optional[int], chunker = str) -> str:
        pass
