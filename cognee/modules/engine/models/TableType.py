from cognee.infrastructure.engine import DataPoint


class TableType(DataPoint):
    name: str
    description: str

    metadata: dict = {"index_fields": ["name"]}
