from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.TableType import TableType
from typing import Optional


class TableRow(DataPoint):
    name: str
    is_a: Optional[TableType] = None
    description: str
    ontology_valid: bool = False
    properties: str

    metadata: dict = {"index_fields": ["properties"]}
