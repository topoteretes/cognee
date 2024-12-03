from cognee.infrastructure.engine import DataPoint
from uuid import UUID

class Document(DataPoint):
    type: str
    name: str
    raw_data_location: str
    metadata_id: UUID

    def read(self, chunk_size: int) -> str:
        pass
