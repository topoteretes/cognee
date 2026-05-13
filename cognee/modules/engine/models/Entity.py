from typing import List, Optional

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.EntityType import EntityType


class Entity(DataPoint):
    name: str
    is_a: Optional[EntityType] = None
    description: str
    relations: List[tuple] = []
    metadata: dict = {"index_fields": ["name"]}
