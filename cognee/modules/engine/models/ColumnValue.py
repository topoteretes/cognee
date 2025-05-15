from cognee.infrastructure.engine import DataPoint


class ColumnValue(DataPoint):
    name: str
    description: str
    properties: str

    metadata: dict = {"index_fields": ["properties"]}
