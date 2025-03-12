from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.EntityType import EntityType
from typing import Optional


class Entity(DataPoint):
    name: str
    is_a: Optional[EntityType] = None
    description: str
    ontology_valid: bool = False

    metadata: dict = {"index_fields": ["name"]}
