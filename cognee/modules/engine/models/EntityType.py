from cognee.infrastructure.engine import DataPoint


class EntityType(DataPoint):
    name: str
    description: str

    metadata: dict = {"index_fields": ["name"]}
