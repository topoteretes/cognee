from cognee.infrastructure.engine import DataPoint


class TableType(DataPoint):
    name: str
    description: str
    ontology_valid: bool = False

    metadata: dict = {"index_fields": ["name"]}
