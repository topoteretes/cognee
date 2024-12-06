from cognee.infrastructure.engine import DataPoint

class EntityType(DataPoint):
    __tablename__ = "entity_type"
    name: str
    type: str
    description: str

    _metadata: dict = {
        "index_fields": ["name"],
    }
