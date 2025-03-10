from cognee.infrastructure.engine import DataPoint


class EntityType(DataPoint):
    name: str
    description: str
    ontology_valid: bool = False

    metadata: dict = {"index_fields": ["name"]}
