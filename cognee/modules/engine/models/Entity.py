from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from .EntityType import EntityType

class Entity(DataPoint):
    name: str
    is_a: EntityType
    description: str
    mentioned_in: DocumentChunk
    _metadata: dict = {
        "index_fields": ["name"],
    }
