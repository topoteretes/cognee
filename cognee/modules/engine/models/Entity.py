from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.EntityType import EntityType


class Entity(DataPoint):
    __tablename__ = "entity"
    name: str
    is_a: EntityType
    description: str
    pydantic_type: str = "Entity"

    _metadata: dict = {"index_fields": ["name"], "type": "Entity"}
