from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.EntityType import EntityType
from typing import Optional


class Entity(DataPoint):
    name: str
    is_a: Optional[EntityType] = None
    description: str

    metadata: dict = {"index_fields": ["name"]}
