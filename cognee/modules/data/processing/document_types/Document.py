from cognee.infrastructure.engine import DataPoint
from uuid import UUID

class Document(DataPoint):
    type: str
    name: str
    raw_data_location: str
    metadata_id: UUID
    mime_type: str

    def read(self, chunk_size: int, chunker = str) -> str:
        pass
