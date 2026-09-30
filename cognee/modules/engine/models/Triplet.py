from cognee.infrastructure.engine import DataPoint


class Triplet(DataPoint):
    text: str
    from_node_id: str
    to_node_id: str

    metadata: dict = {"index_fields": ["text"]}
