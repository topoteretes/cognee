from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.EntityType import EntityType
from typing import Optional


class TableRow(DataPoint):
    name: str
    text: str
    is_a: Optional[EntityType] = None
    description: str
    ontology_valid: bool = False
    properties: str

    metadata: dict = {"index_fields": ["properties"]}
