from cognee.infrastructure.engine import DataPoint


class EdgeType(DataPoint):
    relationship_name: str
    number_of_edges: int

    metadata: dict = {"index_fields": ["relationship_name"]}
