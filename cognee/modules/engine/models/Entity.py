from typing import Union

from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.engine.models.EntityType import EntityType
from cognee.shared.CodeGraphEntities import Repository


class Entity(DataPoint):
    __tablename__ = "entity"
    name: str
    is_a: EntityType
    description: str
    mentioned_in: Union[DocumentChunk, Repository]
    _metadata: dict = {
        "index_fields": ["name"],
    }
