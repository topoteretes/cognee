from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.EntityType import EntityType


class Entity(DataPoint):
    name: str
    is_a: EntityType
    description: str

    metadata: dict = {"index_fields": ["name"]}
