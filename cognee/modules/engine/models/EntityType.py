from typing import Union

from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.shared.CodeGraphEntities import Repository


class EntityType(DataPoint):
    __tablename__ = "entity_type"
    name: str
    type: str
    description: str
    exists_in: Union[DocumentChunk, Repository]
    _metadata: dict = {
        "index_fields": ["name"],
    }
