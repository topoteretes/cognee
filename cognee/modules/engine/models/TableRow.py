from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.TableType import TableType
from typing import Optional


class TableRow(DataPoint):
    name: str
    is_a: Optional[TableType] = None
    description: str
    properties: str

    metadata: dict = {"index_fields": ["properties"]}
