from cognee.infrastructure.engine import DataPoint


class EdgeInstance(DataPoint):
    text: str
    relationship_name: str
    source_node_id: str
    target_node_id: str

    metadata: dict = {"index_fields": ["text"]}
