from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.TableType import TableType


class TableRow(DataPoint):
    name: str
    description: str
    properties: str

    metadata: dict = {"index_fields": ["properties"]}
