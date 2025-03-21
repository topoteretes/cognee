from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.EntityType import EntityType
from typing import Optional


class TableRow(DataPoint):
    name: str
    is_a: Optional[EntityType] = None
    description: str
    ontology_valid: bool = False
    properties: dict

    metadata: dict = {"index_fields": ["properties"]}
