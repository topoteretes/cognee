from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk

class EntityType(DataPoint):
    name: str
    type: str
    description: str
    exists_in: DocumentChunk
    _metadata: dict = {
        "index_fields": ["name"],
    }
